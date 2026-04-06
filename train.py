# ===================================================================
#  Training loop
# ===================================================================
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from regularisers import _gradient_magnitudes
from model import GaussianMixtureField
from loss import loss_volume, loss_mip

from sampling import (sample_points_from_volume, sample_points_with_neighbors)
from utils import (setup_logger, weight_schedule, compute_tau_schedule, mip_teacher_z, sample_pixels_from_mip)

def train(
    field: GaussianMixtureField,
    vol: np.ndarray,
    cfg: dict,
    device: str,
    log_dir: str,
) -> GaussianMixtureField:
    logger = setup_logger(log_dir)
    field = field.to(torch.device(device))
    field.train()

    tc = cfg["training"]  # shorthand
    mode = tc.get("mode", "volume").lower()
    steps = int(tc.get("steps", 20000))
    lr = float(tc.get("learning_rate", 1e-2))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, field.parameters()), lr=lr)
    lr_min_frac = float(tc.get("lr_min_fraction", 0.01))  # decay to 1% of initial LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * lr_min_frac
    )

    use_amp = bool(tc.get("mixed_precision", False)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -- sampling ---------------------------------------------------------
    vol_pts = int(tc.get("vol_points_per_step", 8192))
    iwt = bool(tc.get("vol_intensity_weighted", True))

    use_grad = bool(tc.get("use_grad_loss", True))
    delta_vox = int(tc.get("grad_delta_vox", 1))

    # -- loss weights -----------------------------------------------------
    w_grad = float(tc.get("lambda_grad", 0.3))
    w_tube = float(tc.get("lambda_tube", 1e-4))
    w_cross = float(tc.get("lambda_cross", 1e-4))
    w_scale = float(tc.get("lambda_scale", 5e-4))
    scale_target = tc.get("scale_target", 0.03)
    if scale_target is not None:
        scale_target = float(scale_target)

    # -- MIP --------------------------------------------------------------
    mip_px = int(tc.get("mip_pixels_per_step", 4096))
    n_z = int(tc.get("mip_z_samples", 128))
    tau_s = float(tc.get("tau_start", 0.08))
    tau_e = float(tc.get("tau_end", 0.02))
    mip_img = mip_teacher_z(vol)

    # -- amplitude clamping ------------------------------------------------
    clamp_amp = bool(tc.get("clamp_amplitudes", True))
    la_min = float(tc.get("log_amp_min", math.log(1e-4)))   # exp(-9.2) ≈ 0.0001
    la_max = float(tc.get("log_amp_max", math.log(1.0)))    # exp(0) = 1.0
    if clamp_amp:
        logger.info(f"Amplitude clamp=[{la_min:.3f}, {la_max:.3f}]  (amp=[{math.exp(la_min):.5f}, {math.exp(la_max):.4f}])")

    # -- gradient clipping -------------------------------------------------
    grad_clip = float(tc.get("grad_clip_norm", 1.0))  # 0 = disabled
    if grad_clip > 0:
        logger.info(f"Gradient clipping: max_norm={grad_clip}")

    # -- scale clamping ---------------------------------------------------
    do_clamp = bool(tc.get("clamp_scales", True))
    ls_min = float(tc.get("log_scale_min", math.log(5e-4)))   # ~0.0005
    ls_max = float(tc.get("log_scale_max", math.log(0.3)))    # ~0.3 in normalised coords

    # Sanity check: warn if init_scale would be clamped immediately
    init_ls = math.log(float(cfg["model"].get("init_scale", 0.05)))
    if do_clamp and (init_ls < ls_min or init_ls > ls_max):
        logger.warning(
            f"init_scale log={init_ls:.3f} outside clamp range [{ls_min:.3f}, {ls_max:.3f}] "
            f"— Gaussians will be clamped at init! Consider widening clamp range."
        )

    # -- AABB hard clamp --------------------------------------------------
    aabb_hard = bool(tc.get("enforce_aabb_hard", False))

    # -- densify ----------------------------------------------------------
    dens_on = bool(tc.get("densify_enabled", False))
    dens_from = int(tc.get("densify_from_iter", 500))
    dens_until = int(tc.get("densify_until_iter", 25000))
    dens_every = int(tc.get("densify_interval", 100))
    dens_grad = float(tc.get("densify_grad_threshold", 1.5e-4))
    dens_minamp = float(tc.get("densify_min_amplitude", 5e-4))
    dens_maxsc = float(tc.get("densify_max_scale", 0.8))
    dens_split = float(tc.get("densify_split_scale_threshold", 0.05))
    dens_aabb = bool(tc.get("densify_enforce_aabb", True))
    dens_lr_fac = float(tc.get("densify_lr_factor", 0.2))
    dens_lr_warm = int(tc.get("densify_lr_warmup_steps", 25))
    dens_maxK = int(tc.get("max_gaussians", 20000))
    dens_max_clones = int(tc.get("densify_max_clones_per_step", 0))  # 0 = unlimited
    dens_cooldown = int(tc.get("densify_cooldown_evals", 5))  # skip ES checks after densify
    last_dens = -(10**9)

    # EMA accumulator for mean gradient magnitudes (more stable than single-step)
    grad_accum = torch.zeros(field.num_gaussians, device=device)
    grad_count = 0

    # -- progressive mode -------------------------------------------------
    prog_split = float(tc.get("progressive_split_frac", 0.3))

    # -- prepare GPU volume -----------------------------------------------
    if device != "cuda":
        raise RuntimeError(
            "This script expects CUDA.  Add a CPU fallback path if needed."
        )
    vol_gpu = torch.from_numpy(vol).float().to(device)

    # -- logging ----------------------------------------------------------
    logger.info(f"Device={device}  Mode={mode}  Steps={steps}  LR={lr}  AMP={use_amp}")
    logger.info(f"Volume {vol.shape}  MIP {mip_img.shape}")
    logger.info(
        f"Losses: grad={use_grad}(w={w_grad}), tube={w_tube}, cross={w_cross}, "
        f"scale={w_scale}(target={scale_target})"
    )
    logger.info("Compare space: intensity-only")
    logger.info(f"Scale clamp={do_clamp}  [{ls_min:.3f}, {ls_max:.3f}]")
    logger.info(f"Densify={dens_on}  from={dens_from} until={dens_until} every={dens_every} max_K={dens_maxK}")
    logger.info(f"K={field.num_gaussians}")

    # -- timing -----------------------------------------------------------
    timings: dict[str, list[float]] = {
        "sample": [], "vol_fwd": [], "mip_fwd": [], "backward": [], "optim": [],
    }

    # -- early stopping --------------------------------------------------
    es_on = bool(tc.get("early_stopping", False))
    es_patience = int(tc.get("early_stopping_patience", 20))
    es_min_delta = float(tc.get("early_stopping_min_delta", 0.01))  # dB
    es_best_psnr = -float('inf')
    es_no_improve = 0
    es_best_path = None
    if es_on:
        logger.info(f"Early stopping: patience={es_patience} evals, min_delta={es_min_delta} dB")

    save_path = tc.get("save_path")

    best_total = float('inf')
    last_psnr = None
    last_densify_stats = None
    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        # --- Step-0 diagnostic: verify initialization isn't dead ---
        if step == 0:
            with torch.no_grad():
                test_pts, test_vals = sample_points_from_volume(
                    vol_gpu, min(1024, vol_pts),
                    intensity_weighted=iwt, cache_key="vol" if iwt else None,
                )
                test_pred = field(test_pts)
                test_cmp = test_pred
                gt_cmp = test_vals
                pred_mean = float(test_cmp.mean())
                pred_max = float(test_cmp.max())
                pred_nonzero = float((test_cmp > 1e-6).float().mean())
                gt_mean = float(gt_cmp.mean())
                scales_now = torch.exp(field.log_scales)
                amps_now = torch.exp(field.log_amplitudes)
                logger.info(
                    f"INIT CHECK(intensity-space): pred mean={pred_mean:.6f} max={pred_max:.6f} "
                    f"nonzero={pred_nonzero:.1%} | gt mean={gt_mean:.6f} | "
                    f"scale mean={float(scales_now.mean()):.6f} "
                    f"amp mean={float(amps_now.mean()):.6f}"
                )
                if pred_mean < 1e-6:
                    logger.warning(
                        "⚠ Model output is near-zero at init! "
                        "Gaussians are too small or amplitudes too low. "
                        "Increase init_scale or init_amplitude."
                    )

        t_frac = step / max(1, steps - 1)
        tau = compute_tau_schedule(tau_s, tau_e, t_frac)
        wv, wm = weight_schedule(cfg, step, steps)

        # restore LR after densify warmup (let cosine scheduler take over)
        if step - last_dens == dens_lr_warm:
            # Recreate scheduler at correct position with base LR
            for pg in optimizer.param_groups:
                pg.setdefault('initial_lr', lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=lr * lr_min_frac,
                last_epoch=step,
            )

        optimizer.zero_grad(set_to_none=True)

        # current mode (progressive switches volume → hybrid)
        cur = mode
        if mode == "progressive":
            cur = "volume" if t_frac < prog_split else "hybrid"

        total_loss = torch.zeros((), device=device)
        losses: dict[str, float] = {"tau": tau, "wv": wv, "wm": wm}

        amp_ctx = torch.amp.autocast("cuda") if use_amp else torch.nullcontext()
        with amp_ctx:
            # ---------- volume branch ----------
            if cur in ("volume", "hybrid"):
                t0 = time.time()
                if use_grad:
                    (
                        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
                    ) = sample_points_with_neighbors(
                        vol_gpu, vol_pts, delta_vox=delta_vox,
                        intensity_weighted=iwt,
                        cache_key="vol" if iwt else None,
                    )
                else:
                    x, v = sample_points_from_volume(
                        vol_gpu, vol_pts, intensity_weighted=iwt,
                        cache_key="vol" if iwt else None,
                    )
                    x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None

                if device == "cuda":
                    torch.cuda.synchronize()
                timings["sample"].append(time.time() - t0)

                t0 = time.time()
                lv, pv = loss_volume(
                    field, x, v,
                    x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
                    w_grad=w_grad if use_grad else 0.0,
                    w_tube=w_tube,
                    w_cross=w_cross,
                    w_scale=w_scale,
                    scale_target=scale_target,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                timings["vol_fwd"].append(time.time() - t0)

                total_loss = total_loss + wv * lv
                for k, vv in pv.items():
                    losses[f"v_{k}"] = float(vv.detach())

            # ---------- MIP branch ----------
            if cur in ("mip", "hybrid"):
                t0 = time.time()
                xy, mt = sample_pixels_from_mip(mip_img, mip_px)
                xy, mt = xy.to(device), mt.to(device)

                lm, pm = loss_mip(
                    field, xy, mt, n_z, tau,
                    w_tube=w_tube, w_cross=w_cross,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                timings["mip_fwd"].append(time.time() - t0)

                total_loss = total_loss + wm * lm
                for k, vv in pm.items():
                    losses[f"m_{k}"] = float(vv.detach())

        # ---------- backward + step ----------
        t0 = time.time()
        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(field.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(field.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        if device == "cuda":
            torch.cuda.synchronize()
        timings["backward"].append(time.time() - t0)

        # ---------- post-step constraints ----------
        if aabb_hard:
            field.apply_aabb_clamp()
        if do_clamp:
            field.clamp_log_scales_(ls_min, ls_max)
        if clamp_amp:
            field.clamp_log_amplitudes_(la_min, la_max)

        losses["total"] = float(total_loss.detach())

        # ---------- accumulate gradient magnitudes for densify ----------
        if dens_on and dens_from <= step <= dens_until:
            g = _gradient_magnitudes(field.means)
            # Handle size mismatch after densify (grad_accum was reset)
            if g.shape[0] == grad_accum.shape[0]:
                grad_accum += g
                grad_count += 1
            else:
                grad_accum = g.clone()
                grad_count = 1

        # ---------- densify / prune ----------
        if (
            dens_on
            and dens_from <= step <= dens_until
            and step % dens_every == 0
            and step > 0
            and grad_count > 0
        ):
            # Use average gradient over accumulation window
            avg_grad = grad_accum / max(grad_count, 1)

            # Log gradient stats for diagnostics
            gmax = float(avg_grad.max())
            gmean = float(avg_grad.mean())
            gmed = float(avg_grad.median())
            above = int((avg_grad > dens_grad).sum())
            logger.info(
                f"GradStats@{step}: max={gmax:.6f} mean={gmean:.6f} "
                f"median={gmed:.6f} above_thresh={above}/{field.num_gaussians} "
                f"(thresh={dens_grad:.6f})"
            )

            # Temporarily inject averaged gradients for densify decision
            old_grad = field.means.grad
            field.means.grad = avg_grad.unsqueeze(-1).expand_as(field.means).contiguous()

            stats = field.densify_and_prune(
                grad_threshold=dens_grad,
                min_amplitude=dens_minamp,
                max_scale=dens_maxsc,
                split_scale_threshold=dens_split,
                enforce_aabb=dens_aabb,
                max_gaussians=dens_maxK,
                max_clones=dens_max_clones,
            )
            last_densify_stats = stats
            logger.info(f"Densify@{step}: {stats}")

            # Restore original grad (will be None on new params anyway)
            if old_grad is not None and old_grad.shape == field.means.shape:
                field.means.grad = old_grad

            # Reset accumulator for new Gaussian set
            grad_accum = torch.zeros(field.num_gaussians, device=device)
            grad_count = 0

            last_dens = step
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, field.parameters()), lr=lr * dens_lr_fac
            )
            # Recreate scheduler at current position so cosine decay continues
            for pg in optimizer.param_groups:
                pg.setdefault('initial_lr', lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=lr * lr_min_frac,
                last_epoch=step,
            )
            if use_amp:
                scaler = torch.amp.GradScaler("cuda")
            if device == "cuda":
                torch.cuda.empty_cache()

        # ---------- progress bar ----------
        if "total" in losses:
            best_total = min(best_total, losses["total"])
        d = {"total": f"{losses.get('total', 0):.4g}"}
        if last_psnr is not None:
            d["psnr"] = f"{last_psnr:.2f}dB"
        if last_densify_stats is not None:
            d["K"] = f"{last_densify_stats.get('new', field.num_gaussians)}"
            pruned = last_densify_stats.get('pruned', 0)
            split = last_densify_stats.get('split', 0)
            if pruned > 0 or split > 0:
                d["prune/split"] = f"{pruned}↓{split}↑"
        pbar.set_postfix(d)

        log_every = int(tc.get("log_every", 50))
        if (step + 1) % log_every == 0:
            logger.info(f"Step {step+1}: K={field.num_gaussians}  best_total={best_total:.6g}  {losses}")

        # ---------- PSNR evaluation ----------
        psnr_every = int(tc.get("psnr_eval_every", 500))
        if (step + 1) % psnr_every == 0:
            with torch.no_grad():
                # Sample 100K random voxels (uniform, not intensity-weighted) for unbiased PSNR
                eval_n = min(100_000, vol_gpu.numel())
                Z, Y, X = vol_gpu.shape
                idx = torch.randint(0, vol_gpu.numel(), (eval_n,), device=device)
                ez = idx // (Y * X)
                ey = (idx % (Y * X)) // X
                ex = idx % X
                exn = (ex.float() / max(X - 1, 1)) * 2 - 1
                eyn = (ey.float() / max(Y - 1, 1)) * 2 - 1
                ezn = (ez.float() / max(Z - 1, 1)) * 2 - 1
                eval_pts = torch.stack([exn, eyn, ezn], dim=1)
                eval_gt = vol_gpu[ez, ey, ex]
                # Chunked evaluation to avoid OOM
                eval_pred_chunks = []
                chunk_sz = 16384
                for ci in range(0, eval_n, chunk_sz):
                    eval_pred_chunks.append(field(eval_pts[ci:ci+chunk_sz]))
                eval_pred_raw = torch.cat(eval_pred_chunks)
                eval_pred = eval_pred_raw
                eval_gt_cmp = eval_gt
                mse = F.mse_loss(eval_pred, eval_gt_cmp).item()
                mae = F.l1_loss(eval_pred, eval_gt_cmp).item()
                psnr = -10 * math.log10(max(mse, 1e-12))
                last_psnr = psnr
                logger.info(f"PSNR@{step+1} (intensity-space): {psnr:.2f} dB  (MSE={mse:.6f}, MAE={mae:.6f}, eval_pts={eval_n})")

                # Early stopping check (skip during cooldown after densify)
                steps_since_dens = step - last_dens
                evals_since_dens = steps_since_dens // psnr_every
                in_cooldown = dens_on and (evals_since_dens < dens_cooldown)
                if es_on and in_cooldown:
                    logger.info(f"Early stopping: cooldown ({evals_since_dens+1}/{dens_cooldown} evals after densify)")
                elif es_on:
                    if psnr > es_best_psnr + es_min_delta:
                        es_best_psnr = psnr
                        es_no_improve = 0
                        # Save best checkpoint
                        if save_path:
                            es_best_path = save_path.replace(".pt", "_best.pt")
                            os.makedirs(os.path.dirname(es_best_path) or ".", exist_ok=True)
                            torch.save(field.state_dict(), es_best_path)
                            logger.info(f"New best PSNR: {es_best_psnr:.2f} dB → {es_best_path}")
                    else:
                        es_no_improve += 1
                        logger.info(f"Early stopping: no improvement {es_no_improve}/{es_patience} (best={es_best_psnr:.2f} dB)")
                    if es_no_improve >= es_patience:
                        logger.info(f"Early stopping triggered at step {step+1} (best PSNR: {es_best_psnr:.2f} dB)")
                        break

        # ---------- checkpoint ----------
        ckpt_every = int(tc.get("checkpoint_interval", 1000))
        if save_path and (step + 1) % ckpt_every == 0:
            base = save_path.replace(".pt", "")
            ckpt = f"{base}_step{step+1}.pt"
            os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
            torch.save(field.state_dict(), ckpt)
            logger.info(f"Checkpoint → {ckpt}")

    # -- timing report ----------------------------------------------------
    # -- load best early-stopping checkpoint if available ---------------
    if es_on and es_best_path and os.path.exists(es_best_path):
        field.load_state_dict(torch.load(es_best_path, weights_only=True))
        logger.info(f"Loaded best checkpoint (PSNR {es_best_psnr:.2f} dB) from {es_best_path}")

    logger.info("=" * 60)
    logger.info("TIMING  (mean ± std  ms)")
    logger.info("=" * 60)
    total_ms = 0.0
    for key in ["sample", "vol_fwd", "mip_fwd", "backward"]:
        if timings[key]:
            arr = np.array(timings[key]) * 1000
            total_ms += arr.mean()
            logger.info(f"  {key:12s}: {arr.mean():7.2f} ± {arr.std():5.2f}")
    if total_ms > 0:
        logger.info(f"  {'TOTAL':12s}: {total_ms:7.2f} ms/step  ({1000/total_ms:.1f} it/s)")
    logger.info("=" * 60)
    logger.info("Training finished.")
    return field
