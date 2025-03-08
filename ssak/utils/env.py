import os
import sys

from .misc import get_cache_dir

DISABLE_GPU = False
REQUIRED_GPU = []

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# For RuntimeError: CUDA out of memory. Tried to allocate 1.03 GiB (GPU 0; 11.93 GiB total capacity; 7.81 GiB already allocated; 755.69 MiB free; 10.59 GiB reserved in total by PyTorch)
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def _set_visible_gpus(s):
    global DISABLE_GPU, REQUIRED_GPU
    if isinstance(s, str):
        if s.lower() == "auto":
            # Choose the GPU with the most free memory
            from ssak.utils.monitoring import get_num_gpus, vram_free

            # GPUs sorted by decreasing free memory
            gpus = list(reversed(sorted(range(get_num_gpus()), key=vram_free)))
            s = str(gpus[0]) if len(gpus) else ""
        if s.lower() == "none":
            s = ""
        return _set_visible_gpus(s.split(",") if s else [])
    if isinstance(s, list):
        s = [int(si) for si in s]
        REQUIRED_GPU = list(range(len(s)))
        s = ",".join([str(si) for si in s])
    if not s:
        DISABLE_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"] = s


has_set_gpu = False
for i, arg in enumerate(sys.argv[1:]):
    arg = arg.lower()
    if arg in ["--gpus", "--gpu"]:
        _set_visible_gpus(sys.argv[i + 2])
        has_set_gpu = True
    elif arg.startswith("--gpus=") or arg.startswith("--gpu="):
        _set_visible_gpus(arg.split("=")[-1])
        has_set_gpu = True
if not has_set_gpu:
    _set_visible_gpus("auto")

# To address the following error when importing librosa
#   RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.9/site-packages/librosa/util/utils.py'
# See https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

# Disable warnings of type "This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if not os.environ.get("HOME"):
    path = os.path.dirname(os.path.abspath(__file__))
    if path.startswith("/home/"):
        os.environ["HOME"] = "/".join(os.environ["HOME"].split("/")[:3])

# Set cache directory
os.environ["HUGGINGFACE_HUB_CACHE"] = get_cache_dir("huggingface/hub")
os.environ["HF_HOME"] = get_cache_dir("huggingface/hub")
os.environ["TRANSFORMERS_CACHE"] = get_cache_dir("huggingface/hub")
import datasets

datasets.config.HF_MODULES_CACHE = get_cache_dir("huggingface/modules")
datasets.config.HF_DATASETS_CACHE = get_cache_dir("huggingface/datasets")
datasets.config.HF_METRICS_CACHE = get_cache_dir("huggingface/metrics")
datasets.config.DOWNLOADED_DATASETS_PATH = get_cache_dir("huggingface/datasets/downloads")

# Importing torch must be done after having set the CUDA-related environment variables
import multiprocessing

import torch


def auto_device():
    return torch.device("cuda:0") if (torch.cuda.is_available() and not DISABLE_GPU) else torch.device("cpu")


def use_gpu():
    if DISABLE_GPU or not torch.cuda.is_available():
        assert REQUIRED_GPU == [], f"GPU required but not available (required GPU: {REQUIRED_GPU})"
        return []
    from ssak.utils.monitoring import get_num_gpus

    num_gpus = get_num_gpus()
    if REQUIRED_GPU:
        assert num_gpus - 1 >= max(REQUIRED_GPU), f"More GPU required than available (required GPU: {REQUIRED_GPU}, available GPU: {list(range(num_gpus))})"
    return REQUIRED_GPU


if use_gpu():
    pass
else:
    # Use maximum number of threads
    torch.set_num_threads(multiprocessing.cpu_count())
