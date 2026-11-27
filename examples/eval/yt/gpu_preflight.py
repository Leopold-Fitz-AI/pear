"""Small BF16 and Triton check under the isolated CUDA compatibility libraries."""

import ctypes
import json

import torch
import transformers
import triton
import triton.language as tl
import vllm


@triton.jit
def add_one(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + ids, ids < N, other=0)
    tl.store(y + ids, value + 1, ids < N)


def main():
    driver = ctypes.c_int()
    ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(driver))
    assert torch.cuda.device_count() == 1
    a = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    assert (a @ a)[0, 0].item() == 32
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(4,)](x, y, 1024, 256)
    torch.cuda.synchronize()
    assert torch.equal(y, x + 1)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "driver_api": driver.value,
                "vllm": vllm.__version__,
                "transformers": transformers.__version__,
                "device": torch.cuda.get_device_name(0),
                "bf16_matmul": True,
                "triton_jit": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
