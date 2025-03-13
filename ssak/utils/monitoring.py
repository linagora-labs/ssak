import os

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import logging
import threading
import time
from datetime import datetime

import numpy as np
import psutil  # RAM monitoring
import py3nvml.py3nvml as pynvml  # VRAM (GPU) monitoring

logging.basicConfig(
    # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    # datefmt="%m/%d/%Y %H:%M:%S",
    # handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(logging.INFO)

"""
This module provides helpers to monitor timing and memory usages.

Example usage (minimal):

```python
from ssak.utils.monitoring import tic, toc, vram_peak, ram_peak
import torch

tic()

print(f"> {ram_peak()=}")
print(f"> {vram_peak()=}")
print(f"> {vram_peak(index=0)=}")

print("Allocating memory...")
x = torch.rand(1000, 1000, device="cpu")
y = torch.rand(1000, 1000, device="cuda:0")

print(f"> {ram_peak()=}")
print(f"> {vram_peak()=}")
print(f"> {vram_peak(index=0)=}")

print(f"> {toc()=}")
```

The above should print something like:

```
INFO:ssak:RAM_CURRENT : 356.95703125 MB
INFO:ssak:RAM_PEAK : 356.95703125 MB
> ram_peak()=356.95703125
INFO:ssak:VRAM_CURRENT : 1/2 NVIDIA GeForce GTX TITAN X: 81 / 12288 MB
INFO:ssak:VRAM_CURRENT : 2/2 NVIDIA GeForce GTX 1080 Ti: 91 / 11264 MB
INFO:ssak:VRAM_PEAK : 172 MB
> vram_peak()=172
INFO:ssak:VRAM_CURRENT : 1/1 NVIDIA GeForce GTX TITAN X: 81 / 12288 MB
INFO:ssak:VRAM_PEAK : 81 MB
> vram_peak(index=0)=81
Allocating memory...
INFO:ssak:RAM_CURRENT : 445.61328125 MB
INFO:ssak:RAM_PEAK : 445.61328125 MB
> ram_peak()=445.61328125
INFO:ssak:VRAM_CURRENT : 1/2 NVIDIA GeForce GTX TITAN X: 204 / 12288 MB
INFO:ssak:VRAM_CURRENT : 2/2 NVIDIA GeForce GTX 1080 Ti: 93 / 11264 MB
INFO:ssak:VRAM_PEAK : 297 MB
> vram_peak()=297
INFO:ssak:VRAM_CURRENT : 1/1 NVIDIA GeForce GTX TITAN X: 204 / 12288 MB
INFO:ssak:VRAM_PEAK : 204 MB
> vram_peak(index=0)=204
INFO:ssak:TIMING : took 0.7578799724578857 sec
> toc()=0.7578799724578857
```
"""

############################################
# Helpers for timing


def tic(name=""):
    """start clock
    Args:
        name: name of the clock
    """
    global TIC
    TIC[name] = time.time()


def toc(name="", stream=None, verbose=True, log_mem_usage=False, total=False):
    """end clock and returns time elapsed since the last tic
    Args:
        name: name of the clock
        log_mem_usage: if True, log GPU memory usage
    """
    global TIC
    t = time.time() - TIC.get(name, TIC[""])
    TIMES[name] = TIMES.get(name, 0) + t
    if total:
        t = TIMES[name]
    s = f"TIMING{_name_to_suffix(name)} : {t:.3f} sec"
    if stream:
        print(s, file=stream)
    if verbose:
        logger.info(s)
        if log_mem_usage:
            vram_usage(name, ignore_errors=True)
            vram_peak(name)
    return t


############################################
# Helpers to monitor GPU and VRAM memory


def get_num_gpus(ignore_errors=False):
    """
    Returns the number of GPUs available
    """
    global ALL_GPU_INDICES
    if ALL_GPU_INDICES is not None:
        return len(ALL_GPU_INDICES)
    try:
        pynvml.nvmlInit()  # Can throw pynvml.NVMLError_DriverNotLoaded if driver problem
    except pynvml.NVMLError_DriverNotLoaded:
        import torch

        if torch.cuda.is_available():
            raise RuntimeError("CUDA is available but pynvml.NVMLError_DriverNotLoaded. This is probably because you are using a conda environment. Try to install nvidia-smi in the conda environment.")
        return 0
    except Exception as unexpected_error:
        if ignore_errors:
            return 0
        raise unexpected_error
    num_gpus = pynvml.nvmlDeviceGetCount()
    ALL_GPU_INDICES = list(range(num_gpus))
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        ALL_GPU_INDICES = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    return len(ALL_GPU_INDICES)


def has_gpu():
    """
    Returns True if GPU is available
    """
    return get_num_gpus() > 0


def vram_usage(name="", index=None, ignore_errors=False, verbose=True, stream=None, minimum=10):
    """
        Returns the VRAM usage (GPU memory) and logs it (with logger.info).

    Args:
        name: str
            an arbitrary name for this measure (that will be used in the log). Can be left empty for simple usage.
        index: list or int or None
            GPU index or list of indices (if None, all available GPUs are considered)
        ignore_errors: bool
            Do not raise errors if GPU is not available
        verbose: bool
            Use false to disable logging
        stream: stream (with .write() and .flush() methods)
            a stream to write the log to
        minimum: int or float
            Minimum memory usage to report the mem usage (in MiB per GPU)

    Returns:
        Total memory usage in MiB
    """
    if verbose is None:
        verbose = stream == None
    summemused = 0
    indices = range(get_num_gpus(ignore_errors=ignore_errors))
    if indices:
        indices = ALL_GPU_INDICES
    if index is None:
        pass
    elif isinstance(index, int):
        # assert index in indices, f"Got index {index} but only {indices} GPUs available"
        # indices = [index]
        assert index < len(indices), f"Got index {index} but only {len(indices)} GPUs available"
        indices = [indices[index]]
    else:
        new_indices = []
        for i in index:
            #  assert i in indices, f"Got index {index} but only {indices} GPUs available"
            assert i < len(indices), f"Got index {i} but only {len(indices)} GPUs available"
            new_indices.append(indices[i])
        indices = new_indices
    for i, igpu in enumerate(indices):
        handle = _get_gpu_handle(igpu)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpuname = pynvml.nvmlDeviceGetName(handle)
        # use = pynvml.nvmlDeviceGetUtilizationRates(handle) # This info does not seem to be reliable
        memused = info.used / 1024**2
        memtotal = info.total / 1024**2
        if memused >= minimum:  # There is always a residual GPU memory used (1 or a few MB). Less than 10 MB usually means nothing.
            summemused += memused
            s = f"VRAM_CURRENT{_name_to_suffix(name)} : {i+1}/{len(indices)} {gpuname} (max {memtotal:.0f} MB): {memused:.0f} MB"
            if verbose:
                logger.info(s)
            if stream is not None:
                stream.write(f"{s}\n")
                stream.flush()

    return summemused


def vram_peak(name="", index=None, ignore_errors=False, verbose=True, **kwargs):
    """
    Measures and returns peak VRAM usage (maximum GPU memory) and logs it (with logger.info).

    See vram_usage() for arguments
    """
    global VRAM_PEAKS
    if ignore_errors and not has_gpu():
        return 0
    key = f"{name}::{index}"
    VRAM_PEAKS[key] = max(VRAM_PEAKS.get(key, 0), vram_usage(name=name, index=index, verbose=verbose, ignore_errors=False, **kwargs))
    if verbose:
        logger.info(f"VRAM_PEAK{_name_to_suffix(name)} : {VRAM_PEAKS[key]} MB")
    return VRAM_PEAKS[key]


def vram_free(index=0):
    """
    Returns the free GPU memory (in MiB) of the GPU at index `index`

    Args:
        index: int
            GPU index
    """
    handle = _get_gpu_handle(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free // 1024**2


def _get_gpu_handle(index=0):
    if not has_gpu():
        raise RuntimeError("No GPU available")
    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except:
        raise RuntimeError(f"Could not access GPU at index {index}")


############################################
# Helpers to monitor RAM memory


def ram_usage(name="", pids=None, verbose=True, stream=None):
    """
        Returns the RAM usage (in MiB) of the current process and all its subprocesses (if pids is None)

    Args:
        name: str
            an arbitrary name for this measure (that will be used in the log). Can be left empty for simple usage.
        pids: list
            list of process IDs (if None, current process and all its subprocesses)
        verbose: bool
            Use false to disable logging
        stream: stream (with .write() and .flush() methods)
            a stream to write the log to
    Returns:
        Total memory usage in MiB

    """
    if pids is None:
        # Get PIDs of the current process and all its subprocesses
        current_pid = get_current_pid()
        pids = [current_pid] + get_subprocess_pids(current_pid)
    ram = sum([psutil.Process(pid=p).memory_info().rss / (1024 * 1024) for p in pids])
    if verbose or stream:
        s = f"RAM_CURRENT{_name_to_suffix(name)} : {ram:.3f} MB"
        if verbose:
            logger.info(s)
        if stream is not None:
            stream.write(f"{s}\n")
            stream.flush()
    return ram


def ram_peak(name="", pids=None, verbose=True, **kwargs):
    """
    Measures and returns peak RAM usage and logs it (with logger.info).

    See ram_usage() for arguments
    """
    global RAM_PEAKS
    key = f"{name}::{pids}"
    RAM_PEAKS[key] = max(RAM_PEAKS.get(key, 0), ram_usage(name=name, pids=pids, verbose=verbose, **kwargs))
    if verbose:
        logger.info(f"RAM_PEAK{_name_to_suffix(name)} : {RAM_PEAKS[key]:.3f} MB")
    return RAM_PEAKS[key]


def get_current_pid():
    return os.getpid()


def get_subprocess_pids(parent_pid):
    subprocess_pids = []
    parent_process = psutil.Process(parent_pid)
    children = parent_process.children(recursive=True)
    subprocess_pids = [child.pid for child in children]
    return subprocess_pids


def get_pids(python_script_name, **kwargs):
    """
    Returns the list of PIDs of the processes that run the python script `python_script_name`

    See get_processes() for arguments
    """
    return [p.info["pid"] for p in get_processes(python_script_name, **kwargs)]


def get_processes(python_script_name, pids_to_ignore=[]):
    """
    Returns the list of processes that run the python script `python_script_name`
    """
    for p in psutil.process_iter(attrs=[]):
        if p.info["pid"] in pids_to_ignore:
            continue
        cmdline = p.info["cmdline"]
        if len(cmdline) >= 2 and cmdline[0].startswith("python") and cmdline[1] == python_script_name:
            yield p
            for child in p.children(recursive=True):
                yield child


############################################
# Internal

# Global variables
TIC = {}
TIMES = {}
ALL_GPU_INDICES = None
VRAM_PEAKS = {}
RAM_PEAKS = {}


def _name_to_suffix(
    name,
    datefmt=None,  # "%m/%d/%Y|%H:%M:%S"
):
    suffix = ""
    if datefmt:
        formatted_time = datetime.now().strftime(datefmt)
        suffix += f"({formatted_time})"
    if name:
        suffix += f" {name}"
    return suffix


class MonitoringThread(threading.Thread):
    def run(self):
        try:
            super(MonitoringThread, self).run()
        except Exception as e:
            raise Exception("ERROR in Monitoring Thread") from e
            # print("ERROR in Monitoring Thread:", e, file=sys.stderr)
            # sys.exit(1)


class Monitoring:
    """
    This class is used to monitor the hardware usage of the machine while running a script.
    Args:
        output_folder: str
            The folder where the monitoring will be saved
        name: str
            The name of the monitoring (TODO: not used yet)
        interval: float
            The interval (in seconds) at which the monitoring will be done
        device: str or int
            The device to monitor (can be "cpu" or "cuda" or the index of the GPU)
        plot_monitoring: bool
            If True, the monitoring will be plotted
        show_steps_in_plots: bool
            If True, the steps will be shown in the plots

    Example usage:

        ```python
        monitor = Monitoring(output_folder="test", interval=0.2, device="cpu", plot_monitoring=True, show_steps_in_plots=True)
        monitor.start(steps=[str(i) for i in range(10)])
        for i in range(10):
            time.sleep(1)
            monitor.next()
        monitor.stop()
        ```
    """

    def __init__(self, output_folder="", name="", interval=0.25, device="cuda", plot_monitoring=True, show_steps_in_plots=True):
        self.device = device
        self.output_folder = output_folder
        if not name:
            self.name = output_folder
        else:
            self.name = name
        self.interval = interval
        self.show_steps = show_steps_in_plots
        self.will_plot_monitoring = plot_monitoring
        if self.will_plot_monitoring:
            pass

    def _finish_step(self, monitoring, step_values, step=0, start=0):
        for i in step_values:
            if i not in monitoring:
                monitoring[i] = []
            monitoring[i].extend(step_values[i])
        if self.steps and len(self.steps)>0:
            if "steps" not in monitoring:
                monitoring["steps"] = []
            if "steps_end" not in monitoring:
                monitoring["steps_end"] = []
            monitoring["steps"].append(self.steps[step])
            monitoring["steps_end"].append(time.time() - start)
        return monitoring

    def _get_hardware_info(self, handle, start, step_monitoring):
        ram_usage = psutil.virtual_memory().used / 1024**3
        cpu_usage = psutil.cpu_percent()
        time_point = time.time() - start
        if "ram_usage" not in step_monitoring:
            step_monitoring["ram_usage"] = []
        step_monitoring["ram_usage"].append(ram_usage)
        if "cpu_usage" not in step_monitoring:
            step_monitoring["cpu_usage"] = []
        step_monitoring["cpu_usage"].append(cpu_usage)
        if "vram_usage" not in step_monitoring:
            step_monitoring["vram_usage"] = []
        if handle:
            vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_usage = vram_info.used / 1024**3
            gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            step_monitoring["vram_usage"].append(vram_usage)
            if "gpu_usage" not in step_monitoring:
                step_monitoring["gpu_usage"] = []
            step_monitoring["gpu_usage"].append(gpu_usage)
        if "time_points" not in step_monitoring:
            step_monitoring["time_points"] = []
        step_monitoring["time_points"].append(time_point)
        return step_monitoring

    def _save_monitoring(self, monitoring):
        with open(os.path.join(self.output_folder, "monitoring.json"), "w") as f:
            json.dump(monitoring, f, indent=2)

    def _monitor(self):
        if self.device != "cpu":
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        else:
            handle = None
        if os.path.exists(os.path.join(self.output_folder, "monitoring.json")):
            with open(os.path.join(self.output_folder, "monitoring.json")) as f:
                monitoring = json.load(f)
            start = time.time() - monitoring["time_points"][-1]
            if "device" in monitoring and monitoring["device"] != (pynvml.nvmlDeviceGetName(handle) if handle else "cpu"):
                raise ValueError("The device used in the monitoring is different from the one specified in the current monitoring")
        else:
            monitoring = dict()
            monitoring["device"] = pynvml.nvmlDeviceGetName(handle) if handle else "cpu"
            start = time.time()
        step = 0
        step_monitoring = dict()
        while not self.event_stop.is_set() and not self.event_error.is_set():
            if self.event_next.is_set():
                self.event_next.clear()
                monitoring = self._finish_step(monitoring, step_monitoring, step, start)
                self._save_monitoring(monitoring)
                step_monitoring = dict()
                step += 1
            step_monitoring = self._get_hardware_info(handle, start, step_monitoring)
            time.sleep(self.interval)
        if not self.event_error.is_set():
            step_monitoring = self._get_hardware_info(handle, start, step_monitoring)
            monitoring = self._finish_step(monitoring, step_monitoring, step, start)
            if handle:
                monitoring["total_gpu_usage"] = np.trapz(monitoring["gpu_usage"], monitoring["time_points"]) / 100
            self._save_monitoring(monitoring)
            if self.will_plot_monitoring:
                self.plot_monitoring(monitoring, self.output_folder, handle)
        if handle:
            pynvml.nvmlShutdown()

    def start(self, steps=None):
        """
        Start the monitoring (in a separate thread)

        Args:
            steps: list of str
                List of steps to monitor
        """
        self.device = self.device if self.device else 0
        if self.device == "cuda" or self.device == "gpu":
            self.device = 0
        if self.device != "cpu":
            get_num_gpus()
            self.device = ALL_GPU_INDICES[self.device]
        self.event_stop = threading.Event()
        self.event_next = threading.Event()
        self.event_error = threading.Event()
        self.steps = steps
        self.monitoring_thread = MonitoringThread(target=self._monitor)
        self.monitoring_thread.start()

    def next(self):
        """
        Checkpoint the monitoring and goes to the next step
        """
        self.event_next.set()

    def stop(self, error=False):
        """
        Stop the monitoring.
        args:
            error: bool
                If error is True, the monitoring will stop without saving
        """
        if error:
            self.event_error.set()
        else:
            self.event_stop.set()
        self.monitoring_thread.join()

    def plot_hardware(self, values, times, output_folder, ylabel="RAM Usage", lims=None, steps=None):
        import matplotlib.pyplot as plt

        plt.clf()
        plt.plot(times, values, color="skyblue")
        if self.show_steps:
            if isinstance(steps, dict):
                positions = list(steps.values())
                labels = list(steps.keys())
            else:
                positions = steps
                labels = None
            for i in range(len(positions) - 1):
                plt.axvline(x=positions[i], color="red", linestyle="--")
            if labels:
                for i, txt in enumerate(labels):
                    pos = (positions[i] + positions[i - 1]) / 2 if i > 0 else positions[i] / 2
                    plt.text(pos, 0, txt, rotation=90, ha="center", va="top")
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        if lims:
            plt.ylim(lims)
        # plt.title('RAM Usage Over Time')
        plt.xticks(rotation=45, ha="right")
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{ylabel.lower().replace(' ', '_')}.png"), bbox_inches="tight")
        plt.close()

    def plot_monitoring(self, monitoring, output_folder, handle):
        steps = {monitoring["steps"][i]: monitoring["steps_end"][i] for i in range(len(monitoring["steps"]))} if "steps" in monitoring else None
        self.plot_hardware(
            monitoring["ram_usage"],
            monitoring["time_points"],
            output_folder,
            ylabel="RAM Usage",
            steps=steps,
            lims=(0, psutil.virtual_memory().total / 1024**3),
        )
        self.plot_hardware(
            monitoring["cpu_usage"],
            monitoring["time_points"],
            output_folder,
            ylabel="CPU Usage",
            steps=steps,
            lims=(0, 100),
        )
        if handle:
            vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.plot_hardware(
                monitoring["vram_usage"],
                monitoring["time_points"],
                output_folder,
                ylabel="VRAM Usage",
                steps=steps,
                lims=(0, vram_info.total / 1024**3),
            )
            self.plot_hardware(
                monitoring["gpu_usage"],
                monitoring["time_points"],
                output_folder,
                ylabel="GPU Usage",
                steps=steps,
                lims=(0, 100),
            )

    def plot_bench_total_gpu_usage(input_folder, output_folder=None, configs=None, title="Total GPU Usage for Different Benchmarks"):
        import matplotlib.pyplot as plt

        if not output_folder:
            output_folder = input_folder
        if not configs:
            configs = [i for i in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, i))]
        benchs = dict()
        for i in configs:
            if not os.path.exists(os.path.join(input_folder, i, "monitoring.json")):
                continue
            with open(os.path.join(input_folder, i, "monitoring.json")) as f:
                monitoring = json.load(f)
            if "total_gpu_usage" in monitoring:
                benchs[i] = monitoring["total_gpu_usage"]
        plt.clf()
        plt.barh(list(benchs.keys()), list(benchs.values()))
        for i, v in enumerate(list(benchs.values())):
            plt.text(v + 1, i, str(round(v, 1)), fontweight="bold", va="center")
        plt.xlabel("Number of seconds at 100% GPU Usage")
        # plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.savefig(os.path.join(output_folder, title.replace(" ", "_") + ".png"), bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    monitor = Monitoring(output_folder="test", name="test", interval=0.25, device="cpu", show_steps_in_plots=True, plot_monitoring=True)
    monitor.start(steps=[str(i) for i in range(10)])
    for i in range(10):
        time.sleep(1)
        monitor.next()
    monitor.stop()
