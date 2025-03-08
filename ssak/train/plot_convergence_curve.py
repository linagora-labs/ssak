import json
import os
import sys

import matplotlib.pyplot as plt
import transformers

from ssak.utils.env import *
from ssak.utils.misc import commonprefix

# TODO: put that as options
PLOT_LEARNING_RATE = False
PLOT_TRAINING_TIME = False
PLOT_VALIDATION_TIME = False
USE_TENSORBOARD = False
if USE_TENSORBOARD:
    from tensorflow.python.summary.summary_iterator import summary_iterator


def get_log_history(path):
    if path.endswith(".json"):
        return get_log_history_huggingface(path)
    else:
        return get_log_history_speechbrain(path)


def get_log_history_huggingface(path):
    log_history = {}

    key_loss_valid = None
    key_loss_train = None

    initpath = os.path.join(os.path.dirname(os.path.dirname(path)), "init_eval.json")
    if os.path.isfile(initpath):
        with open(initpath) as f:
            d = json.load(f)
        key_loss_valid = "loss/valid" if "loss/valid" in d else "eval_loss"
        key_wer_valid = "WER/valid" if "WER/valid" in d else "eval_wer"
        key_del = "del/valid" if "del/valid" in d else "eval_del"
        key_ins = "ins/valid" if "ins/valid" in d else "eval_ins"
        key_sub = "sub/valid" if "sub/valid" in d else "eval_sub"
        log_history["step"] = [0]
        log_history["loss/train"] = [None]
        log_history["loss/valid"] = [d[key_loss_valid]]
        log_history["WER/valid"] = [d[key_wer_valid]]
        log_history["lr_model"] = [None]
        log_history["del/valid"] = [d.get(key_del)]
        log_history["ins/valid"] = [d.get(key_ins)]
        log_history["sub/valid"] = [d.get(key_sub)]

    with open(path) as f:
        data = json.load(f)
    for d in data["log_history"]:
        step = d["step"]
        steps = log_history.get("step", [])
        if len(steps) == 0 or step > steps[-1]:
            log_history["step"] = steps + [step]

        if key_loss_train is None:
            key_loss_train = "loss/train" if "loss/train" in d else "loss"
        if key_loss_valid is None:
            key_loss_valid = "loss/valid" if "loss/valid" in d else "eval_loss"
            key_wer_valid = "WER/valid" if "WER/valid" in d else "eval_wer"
            key_del = "del/valid" if "del/valid" in d else "eval_del"
            key_ins = "ins/valid" if "ins/valid" in d else "eval_ins"
            key_sub = "sub/valid" if "sub/valid" in d else "eval_sub"

        if key_loss_train in d:
            log_history["loss/train"] = log_history.get("loss/train", []) + [d[key_loss_train]]
        if key_loss_valid in d:
            log_history["loss/valid"] = log_history.get("loss/valid", []) + [d[key_loss_valid]]
        if key_wer_valid in d:
            log_history["WER/valid"] = log_history.get("WER/valid", []) + [d[key_wer_valid]]

        if key_del in d:
            log_history["del/valid"] = log_history.get("del/valid", []) + [d[key_del]]
        if key_ins in d:
            log_history["ins/valid"] = log_history.get("ins/valid", []) + [d[key_ins]]
        if key_sub in d:
            log_history["sub/valid"] = log_history.get("sub/valid", []) + [d[key_sub]]

        if "lr_model" in d:
            log_history["lr_model"] = log_history.get("lr_model", []) + [d["lr_model"]]

    return log_history


def get_log_history_speechbrain(path, only_finished_epochs=False, batch_size=None):
    if not batch_size:
        if "_bs" in path:
            batch_size = int(path.split("_bs")[1].split("_")[0].split("/")[0].split("-")[0])
        else:
            batch_size = 1

    if os.path.isdir(path):
        # Load tensorboard data
        assert USE_TENSORBOARD
        log_files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith("events.out.tfevents")]
        assert len(log_files) == 1, "Found multiple log files: %s" % log_files
        log_file = log_files[0]
        log_history = {}
        last_step = None
        for summary in summary_iterator(log_file):
            step = summary.step
            if step != last_step:
                if len(log_history) == 1:
                    # Skip first step
                    log_history = {}
                log_history["step"] = log_history.get("step", []) + [step]
                last_step = step
            for value in summary.summary.value:
                log_history[value.tag] = log_history.get(value.tag, []) + [value.simple_value]

    else:
        with open(path) as f:
            lines = f.read().splitlines()
        log_history = {}
        last_step = 0
        for line in lines:
            if only_finished_epochs and not simple_get(line, "epoch_finished", eval):
                continue
            step = simple_get(line, "total_samples", int)
            audio_duration = simple_get(line, "total_audio_h", float)
            if step < last_step:
                print(f"Warning: step decreased from {last_step} to {step} in {path}")
                k = batch_size
                step2 = step
                while step2 < last_step:
                    k -= 1
                    assert k > 0
                    step2 = step * batch_size / k
                step = step2
            last_step = step
            log_history["step"] = log_history.get("step", []) + [step]
            log_history["total_audio_h"] = log_history.get("total_audio_h", []) + [audio_duration]
            log_history["loss/train"] = log_history.get("loss/train", []) + [simple_get(line, "train loss")]
            log_history["loss/valid"] = log_history.get("loss/valid", []) + [simple_get(line, "valid loss")]
            wer = simple_get(line, "valid WER")
            log_history["WER/valid"] = log_history.get("WER/valid", []) + [wer]
            log_history["lr_model"] = log_history.get("lr_model", []) + [simple_get(line, "lr_model")]
            train_time = simple_get(line, "train_time_h")
            valid_time = simple_get(line, "valid_time_h")
            log_history["train_time_h"] = log_history.get("train_time_h", []) + [train_time]
            log_history["valid_time_h"] = log_history.get("valid_time_h", []) + [valid_time]

    step_offset = 0
    train_time_offset = 0
    valid_time_offset = 0
    for step, (train_time, valid_time) in enumerate(zip(log_history.get("train_time_h", []), log_history.get("valid_time_h", []))):
        train_time_delta = train_time - train_time_offset
        train_time_offset = train_time
        valid_time_delta = valid_time - valid_time_offset
        valid_time_offset = valid_time
        step_delta = step - step_offset
        step_offset = step
        log_history["train_time_norm"] = log_history.get("train_time_norm", []) + [train_time_delta / (max(1, step_delta) / batch_size)]
        log_history["valid_time_norm"] = log_history.get("valid_time_norm", []) + [valid_time_delta / 60]

    return log_history


def simple_get(line, field, t=float):
    return t(line.split(field + ":")[-1].split()[0].split(",")[0])


def get_monitoring_file(dir):
    if os.path.isfile(dir):
        return dir
    path = transformers.trainer_utils.get_last_checkpoint(dir)
    if path is not None:
        filename = os.path.join(path, "trainer_state.json")
        if os.path.isfile(filename):
            return filename
    else:
        if USE_TENSORBOARD:
            folder = os.path.join(dir, "train_log")
            if os.path.isdir(folder):
                return folder
        filename = os.path.join(dir, "train_log.txt")
        if os.path.isfile(filename):
            return filename
    return None


def boole(b):
    return 1 if b else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training convergence curves.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dirs", help="Directories to plot.", type=str, nargs="+")
    parser.add_argument(
        "--use-time",
        help="Whether to use training time as abscisses (training audio data duration otherwise)",
        default=False,
        action="store_true",
    )
    parser.add_argument("--details", help="Also plot Del/Subs/Ins rates", default=False, action="store_true")
    args = parser.parse_args()

    x_to_legend = {
        "train_time_h": "Training time (h)",
        "total_audio_h": "Training audio data duration (h)",
        "step": "Training steps",
    }
    x_key = None

    dirs = args.dirs
    xkeys = ["train_time_h"] if args.use_time else ["total_audio_h"] + ["step"]

    def get_x(log_history):
        global x_key
        if x_key is not None:
            assert x_key in log_history.keys(), f"{x_key} not in {log_history.keys()}"
            return log_history[x_key]
        for k in xkeys:
            if k in log_history.keys():
                x_key = k
                return log_history[k]
        raise RuntimeError(f"Could not find x keys among {log_history.keys()}")

    dirs = [dir for dir in dirs if get_monitoring_file(dir) is not None]

    prefix = commonprefix(dirs)
    suffix = commonprefix([s[::-1] for s in dirs])

    datas = {}
    for dir in dirs:
        filename = get_monitoring_file(dir)
        if filename is None:
            continue
        if len(suffix):
            dir = dir[len(prefix) : -len(suffix)]
        else:
            dir = dir[len(prefix) :]
        dir = dir.strip("_")
        d = get_log_history(filename)
        if isinstance(d, list):
            for i, di in enumerate(d):
                datas[dir + str(i)] = di
        else:
            datas[dir] = d

    if len(datas) == 0:
        print("No data found. Please specify one or several relevant folders.")
        sys.exit(1)

    colors = ["red", "green", "blue", "black", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta"]
    colors = "brgcmk"

    def get_color(i):
        return colors[i % len(colors)]

    linestyles = ["-", "--", "-.", ":"]

    def get_linestyle(i):
        j = i // len(colors)
        return linestyles[j % len(linestyles)]

    nplots = 2 + boole(PLOT_LEARNING_RATE) + boole(PLOT_TRAINING_TIME) + boole(PLOT_VALIDATION_TIME)
    if args.details:
        nplots += 3

    xmin = min([min(get_x(data)) for data in datas.values()])
    xmax = max([max(get_x(data)) for data in datas.values()])

    best_wer = []
    argbest = []
    for i, (dir, data) in enumerate(datas.items()):
        if "WER/valid" not in data.keys():
            best_wer.append(None)
            best_loss = min([x for x in data["loss/valid"] if x is not None])
            argbest.append(data["loss/valid"].index(best_loss))
        else:
            best_wer.append(min([x for x in data["WER/valid"] if x is not None]))
            argbest.append(data["WER/valid"].index(best_wer[-1]))

    plt.subplot(nplots, 1, 1)
    for i, (dir, data) in enumerate(datas.items()):
        plt.plot(
            get_x(data),
            data["loss/train"],
            get_color(i),
            linestyle=get_linestyle(i),
            label="train" if len(dirs) == 1 else None,
        )
        plt.plot(
            get_x(data),
            data["loss/valid"],
            get_color(i),
            linewidth=3,
            linestyle=get_linestyle(i),
            label="valid" if len(dirs) == 1 else dir,
        )
        plt.plot(get_x(data), data["loss/valid"], get_color(i) + "+", linewidth=3)
        plt.axvline(get_x(data)[argbest[i]], color=get_color(i), linestyle=":")
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.ylabel("loss")

    plt.subplot(nplots, 1, 2)
    for i, (dir, data) in enumerate(datas.items()):
        plt.axvline(get_x(data)[argbest[i]], color=get_color(i), linestyle=":")
        if "WER/valid" not in data.keys():
            continue
        plt.plot(
            get_x(data),
            data["WER/valid"],
            get_color(i),
            linewidth=3,
            linestyle=get_linestyle(i),
            label=f"best: {best_wer[i]:.4g}%",
        )
        plt.plot(get_x(data), data["WER/valid"], get_color(i) + "+", linewidth=3, linestyle=get_linestyle(i))
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.ylabel("WER")

    iplot = 2

    if args.details:
        for what in "sub", "del", "ins":
            iplot += 1
            plt.subplot(nplots, 1, iplot)
            for i, (dir, data) in enumerate(datas.items()):
                d = data.get(what + "/valid", [None])
                if not max([x is not None for x in d]):
                    continue
                plt.plot(
                    get_x(data),
                    d,
                    get_color(i),
                    linewidth=3,
                    linestyle=get_linestyle(i),
                    label=dir if len(dirs) > 1 else None,
                )
                plt.plot(get_x(data), d, get_color(i) + "+", linewidth=3, linestyle=get_linestyle(i))
            plt.xlim(xmin, xmax)
            plt.legend()
            plt.ylabel(what)

    if PLOT_LEARNING_RATE:
        iplot += 1
        plt.subplot(nplots, 1, iplot)
        for i, (dir, data) in enumerate(datas.items()):
            plt.plot(get_x(data), data["lr_model"], get_color(i), label="learning rate" if i == 0 else None)
        # plt.ylabel("Learning Rate")
        plt.xlim(xmin, xmax)
        plt.legend()

    if PLOT_TRAINING_TIME:
        iplot += 1
        plt.subplot(nplots, 1, iplot)
        for i, (dir, data) in enumerate(datas.items()):
            plt.plot(get_x(data), data["train_time_norm"], get_color(i), label="train time (sec/batch)" if i == 0 else None)
        # plt.ylabel("Training Time")
        plt.xlim(xmin, xmax)
        plt.legend()

    if PLOT_VALIDATION_TIME:
        iplot += 1
        plt.subplot(nplots, 1, iplot)
        for i, (dir, data) in enumerate(datas.items()):
            plt.plot(get_x(data), data["valid_time_norm"], get_color(i), label="valid time (min)" if i == 0 else None)
        # plt.ylabel("Validation Time")
        plt.xlim(xmin, xmax)
        plt.legend()

    plt.xlabel(x_to_legend[x_key])
    plt.show()
