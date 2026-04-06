from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _resolve_cuda_home() -> str | None:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and Path(cuda_home).exists():
        return cuda_home

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        nvcc_root = Path(nvcc_path).resolve().parent.parent
        if nvcc_root.exists():
            return str(nvcc_root)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_path = Path(conda_prefix)
        if (conda_path / "bin" / "nvcc").exists():
            return str(conda_path)

    return None


cuda_home = _resolve_cuda_home()
if cuda_home:
    os.environ["CUDA_HOME"] = cuda_home

extra_include_dirs = []
extra_library_dirs = []

if cuda_home:
    cuda_target = Path(cuda_home) / "targets" / "x86_64-linux"
    target_include = cuda_target / "include"
    target_lib = cuda_target / "lib"
    target_lib64 = cuda_target / "lib64"

    if (target_include / "cuda_runtime.h").exists():
        extra_include_dirs.append(str(target_include))
    if target_lib.exists():
        extra_library_dirs.append(str(target_lib))
    if target_lib64.exists():
        extra_library_dirs.append(str(target_lib64))

setup(
    name="gaussian_eval_cuda_forward",
    ext_modules=[
        CUDAExtension(
            name="gaussian_eval_cuda_forward",
            sources=["cuda/gaussian_eval_forward.cu"],
            include_dirs=extra_include_dirs,
            library_dirs=extra_library_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
