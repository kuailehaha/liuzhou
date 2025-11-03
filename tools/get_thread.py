import os
import torch

print("torch version:", torch.__version__)
print("CPU threads before set:", torch.get_num_threads())
get_interop = getattr(torch, "get_num_interop_threads", None)
if get_interop:
    print("interop before set:", get_interop())
else:
    print("interop before set: n/a")

num_cores = os.cpu_count() or 1
torch.set_num_threads(num_cores)
print("CPU threads after set:", torch.get_num_threads())
set_interop = getattr(torch, "set_num_interop_threads", None)
if get_interop and set_interop:
    set_interop(num_cores)
    print("interop threads after set:", get_interop())
else:
    print("interop threads after set: n/a")

print("OMP_NUM_THREADS env:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS env:", os.environ.get("MKL_NUM_THREADS"))