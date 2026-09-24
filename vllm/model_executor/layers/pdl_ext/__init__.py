# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-built CUDA ops for VLLM_PDL_GEMV (torch.ops.flashnext_pdl.*).

- gemv_tma: CUDA decode GEMV (TMA weight ring + mma), VLLM_PDL_GEMV_CUDA.
- gdn_post_conv_mtp: a copy of the fused GDN decode post-conv kernel that triggers
programmatic dependent launch at its start, so out_proj's PDL GEMV streams its
weights while the (few-CTA) recurrent kernel runs. Built once with nvcc into
~/.cache/vllm/flashnext_pdl and loaded from there afterwards.
"""

import os
from functools import cache

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.normpath(os.path.join(_DIR, "..", "..", "..", "..", "csrc"))


@cache
def load() -> bool:
    if hasattr(torch.ops, "flashnext_pdl") and hasattr(torch.ops.flashnext_pdl, "gemv_tma"):
        return True
    from torch.utils.cpp_extension import load as _load

    build_dir = os.path.expanduser("~/.cache/vllm/flashnext_pdl" + os.environ.get("FN_GEMV_TAG", ""))
    os.makedirs(build_dir, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    _load(
        name="flashnext_pdl",
        sources=[os.path.join(_DIR, "bindings.cpp"), os.path.join(_DIR, "gdn_post_conv_pdl.cu"),
                 os.path.join(_DIR, "gemv_tma.cu")],
        extra_include_paths=[_CSRC],
        extra_cflags=["-O3", "-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020B000000000000ULL"],
        extra_cuda_cflags=["-O3", "-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
                           f"-gencode=arch=compute_{arch},code=sm_{arch}", "--use_fast_math",
                           *os.environ.get("FN_GEMV_FLAGS", "").split()],
        build_directory=build_dir,
        is_python_module=False,
        verbose=False,
    )
    return hasattr(torch.ops.flashnext_pdl, "gemv_tma")
