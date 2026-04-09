"""
Real-time MIP splatting viewer — HTTP/polling, no WebSocket.
Works through any proxy (Jupyter, VS Code, SSH).

Usage:
    python viewer.py --ckpt checkpoints/gmf_refined_best.pt
    Open http://localhost:8081 in your browser.
    
"""

import argparse
import io
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, request, send_file
from gsplat.rendering import rasterization


# ---------------------------------------------------------------------------
# Aspect-ratio correction (same as notebook)
# ---------------------------------------------------------------------------
def correct_aspect(means_n, log_scales_n, quats_n, vol_shape):
    Z, Y, X = vol_shape
    aspect = torch.tensor([X/X, Y/X, Z/X], dtype=torch.float32, device=means_n.device)
    means_p = means_n * aspect[None, :]

    q = F.normalize(quats_n, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros(N, 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - w*z);  R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z);      R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y);      R[:, 2, 1] = 2*(y*z + w*x);  R[:, 2, 2] = 1 - 2*(x*x + y*y)

    s  = torch.exp(log_scales_n)
    L  = (torch.diag(aspect)[None] @ R) * s[:, None, :]
    U, sigma, _ = torch.linalg.svd(L)
    U  = U.clone(); U[:, :, -1] *= torch.linalg.det(U)[:, None]

    trace = U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2]
    qp    = torch.zeros(N, 4, dtype=q.dtype, device=q.device)
    qp[:, 0] = torch.sqrt((trace + 1).clamp(min=0)) / 2
    qp[:, 1] = (U[:, 2, 1] - U[:, 1, 2]) / (4 * qp[:, 0].clamp(min=1e-6))
    qp[:, 2] = (U[:, 0, 2] - U[:, 2, 0]) / (4 * qp[:, 0].clamp(min=1e-6))
    qp[:, 3] = (U[:, 1, 0] - U[:, 0, 1]) / (4 * qp[:, 0].clamp(min=1e-6))

    return means_p.contiguous(), sigma.contiguous(), F.normalize(qp, dim=-1).contiguous()


def load_splats(ckpt_path, device, vol_shape=(100, 647, 813)):
    raw  = torch.load(ckpt_path, weights_only=False, map_location=device)
    means, scales, quats = correct_aspect(
        raw["means"].float(), raw["log_scales"].float(), raw["quaternions"].float(), vol_shape
    )
    opacities = torch.exp(raw["log_amplitudes"].float()).squeeze(-1).clamp(0, 1).contiguous()
    colors    = torch.ones(means.shape[0], 1, device=device)
    Z, Y, X = vol_shape
    print(f"Loaded {means.shape[0]:,} Gaussians  aspect: X=1.000 Y={Y/X:.3f} Z={Z/X:.3f}")
    return means, quats, scales, opacities, colors


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
@torch.no_grad()
def render_frame(az_deg, el_deg, radius, W, H, means, quats, scales, opacities, colors, device, eps2d):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    eye = torch.tensor([
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.cos(az),
    ], dtype=torch.float32, device=device)

    forward = F.normalize(-eye, dim=0)
    world_up = torch.tensor([0., 1., 0.], device=device)
    right    = F.normalize(torch.cross(forward, world_up), dim=0)
    up       = torch.cross(right, forward, dim=0)
    R = torch.stack([right, -up, forward], dim=0)
    t = -R @ eye
    E = torch.eye(4, device=device); E[:3, :3] = R; E[:3, 3] = t

    fy = 0.5 * H / math.tan(math.radians(45) / 2)
    K  = torch.tensor([[fy, 0, W/2], [0, fy, H/2], [0, 0, 1]], device=device, dtype=torch.float32)

    rc, _, _ = rasterization(
        means, quats, scales, opacities, colors,
        E[None], K[None], W, H,
        eps2d=eps2d, sh_degree=None, packed=True,
        rasterize_mode="antialiased", render_mode="RGB",
    )
    gray = rc[0, ..., 0].clamp(0, 1).cpu().numpy()
    return (gray * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html>
<head>
<title>MicroGS Viewer</title>
<style>
  body { margin:0; background:#111; display:flex; flex-direction:column;
         align-items:center; justify-content:center; height:100vh; color:#eee;
         font-family:sans-serif; user-select:none; }
  #wrap { position:relative; display:inline-block; }
  #canvas { border:1px solid #333; cursor:grab; display:block; }
  #canvas:active { cursor:grabbing; }
  #fps { position:absolute; top:6px; left:8px; font-size:13px; font-weight:bold;
         color:#0f0; text-shadow:0 0 4px #000, 0 0 2px #000; pointer-events:none; }
  #controls { margin-top:8px; display:flex; gap:16px; font-size:13px; }
  label { display:flex; flex-direction:column; align-items:center; gap:2px; }
</style>
</head>
<body>
<div id="wrap">
  <img id="canvas" width="600" height="450" draggable="false"/>
  <div id="fps"></div>
</div>
<div id="controls">
  <label>Azimuth <input type="range" id="az"  min="0"   max="360" step="1"   value="45"/></label>
  <label>Elevation <input type="range" id="el"  min="-80" max="80"  step="1"   value="20"/></label>
  <label>Radius <input type="range" id="rad" min="0.1" max="5"   step="0.05" value="1.5"/></label>
</div>
<script>
  const img   = document.getElementById('canvas');
  const azEl  = document.getElementById('az');
  const elEl  = document.getElementById('el');
  const radEl = document.getElementById('rad');
  const fpsEl = document.getElementById('fps');

  let drag=false, lastX=0, lastY=0;

  function refresh() {
    const t0 = Date.now();
    const newImg = new Image();
    newImg.onload = () => {
      img.src = newImg.src;
      const ms = Date.now() - t0;
      fpsEl.textContent = `${(1000/ms).toFixed(1)} FPS  (${ms} ms)`;
    };
    newImg.src = `/render?az=${azEl.value}&el=${elEl.value}&rad=${radEl.value}&t=${t0}`;
  }

  azEl.oninput = elEl.oninput = radEl.oninput = refresh;

  img.addEventListener('mousedown', e => { drag=true; lastX=e.clientX; lastY=e.clientY; });
  window.addEventListener('mouseup', () => drag=false);
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    azEl.value = ((+azEl.value + (e.clientX-lastX)) + 360) % 360;
    elEl.value = Math.max(-80, Math.min(80, +elEl.value - (e.clientY-lastY)));
    lastX=e.clientX; lastY=e.clientY;
    refresh();
  });
  img.addEventListener('wheel', e => {
    radEl.value = Math.max(0.1, Math.min(5, +radEl.value + e.deltaY*0.005));
    refresh(); e.preventDefault();
  }, {passive:false});

  refresh();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def make_app(means, quats, scales, opacities, colors, device, eps2d, W, H):
    app = Flask(__name__)
    lock = threading.Lock()

    @app.route("/")
    def index():
        return HTML

    @app.route("/render")
    def render():
        az  = float(request.args.get("az",  45))
        el  = float(request.args.get("el",  20))
        rad = float(request.args.get("rad", 1.5))
        with lock:
            gray = render_frame(az, el, rad, W, H,
                                means, quats, scales, opacities, colors, device, eps2d)
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(gray).save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="checkpoints/gmf_refined_best.pt")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--port",      type=int,   default=8081)
    parser.add_argument("--width",     type=int,   default=600)
    parser.add_argument("--height",    type=int,   default=450)
    parser.add_argument("--eps2d",     type=float, default=0.3)
    parser.add_argument("--vol-shape", type=int,   nargs=3,
                        default=[100, 647, 813], metavar=("Z","Y","X"))
    args = parser.parse_args()

    means, quats, scales, opacities, colors = load_splats(
        args.ckpt, args.device, tuple(args.vol_shape)
    )
    app = make_app(means, quats, scales, opacities, colors,
                   args.device, args.eps2d, args.width, args.height)

    print(f"\nViewer → http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
