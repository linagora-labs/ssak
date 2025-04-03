import os

# To address the following error when importing librosa
#   RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.9/site-packages/librosa/util/utils.py'
# See https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
os.environ["NUMBA_CACHE_DIR"] = "/tmp"


import librosa
import numpy as np
import soxbindings as sox
import torch
import torchaudio

from ssak.utils.misc import (
    run_command,
    suppress_stderr,
    walk_files,
)

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".opus"]


def load_audio(path, start=None, end=None, sample_rate=16_000, mono=True, return_format="array", verbose=False):
    """
    Load an audio file and return the data.

    Parameters
    ----------
    path: str
        path to the audio file
    start: float
        start time in seconds. If None, the file will be loaded from the beginning.
    end: float
        end time in seconds. If None the file will be loaded until the end.
    sample_rate: int
        destination sampling rate in Hz
    mono: bool
        if True, convert to mono
    return_format: str (default: 'array')
        'array': numpy.array
        'torch': torch.Tensor
        'bytes': bytes

    verbose: bool
        if True, print the steps
    """
    assert return_format in ["array", "torch", "bytes", "librosa"]
    if not os.path.isfile(path):
        # Because soxbindings does not indicate the filename if the file does not exist
        raise RuntimeError("File not found: %s" % path)
    # Test if we have read permission on the file
    elif not os.access(path, os.R_OK):
        # os.system("chmod a+r %s" % path)
        raise RuntimeError("Missing reading permission for: %s" % path)

    if verbose:
        print("Loading audio", path, start, end)

    duration = get_audio_duration(path)
    if start is not None and start > duration:
        raise ValueError(f"Start time {start} exceeds the duration of the audio {path}: {duration}")

    must_cut = start or end

    if return_format == "torch" and not must_cut:
        if must_cut:  # This path is super slow and has been disabled
            start = float(start if start else 0)
            sr = torchaudio.info(path).sample_rate
            offset = int(start * sr)
            num_frames = -1
            if end:
                end = float(end)
                num_frames = int((end - start) * sr)
            audio, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_frames)
        else:
            audio, sr = torchaudio.load(path)
    if return_format=="librosa":
        import librosa
        offset = float(start if start else 0)
        duration = None
        if end:
            duration = end - start
        audio, sr = librosa.load(path, offset=offset, duration=duration)
    else:
        with suppress_stderr():
            # stderr could print these harmless warnings:
            # 1/ Could occur with sox.read
            # mp3: MAD lost sync
            # mp3: recoverable MAD error
            # 2/ Could occur with sox.get_info
            # wav: wave header missing extended part of fmt chunk
            if must_cut:  # is not None:
                start = float(start if start else 0)
                sr = sox.get_info(path)[0].rate
                offset = int(start * sr)
                nframes = 0
                if end:  # is not None:
                    end = float(end)
                    nframes = int((end - start) * sr)
                audio, sr = sox.read(path, offset=offset, nframes=nframes)
            else:
                audio, sr = sox.read(path)

        audio = np.float32(audio)

    audio = conform_audio(audio, sr, sample_rate=sample_rate, mono=mono, return_format=return_format, verbose=verbose)

    if verbose:
        print("- Done", path, start, end)

    if sample_rate is None:
        return (audio, sr)
    return audio


def conform_audio(audio, sr, sample_rate=16_000, mono=True, return_format="array", verbose=False):
    if mono:
        if len(audio.shape) == 1:
            pass
        elif len(audio.shape) > 2:
            raise RuntimeError("Audio with more than 2 dimensions not supported")
        elif min(audio.shape) == 1:
            if verbose:
                print("- Reshape to mono")
            audio = audio.reshape(audio.shape[0] * audio.shape[1])
        else:
            if verbose:
                print(f"- Average to mono from shape {audio.shape}")
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            else:
                audio = audio.transpose()
            audio = librosa.to_mono(audio)
    if sample_rate is not None and sr != sample_rate:
        if not isinstance(audio, torch.Tensor):
            if verbose:
                print("- Convert to Torch")
            audio = torch.Tensor(audio)
        if verbose:
            print("- Resample from", sr, "to", sample_rate)
        # We don't use librosa here because there is a problem with multi-threading
        # audio = librosa.resample(audio, orig_sr = sr, target_sr = sample_rate)
        audio = torchaudio.transforms.Resample(sr, sample_rate)(torch.Tensor(audio))

    if return_format == "torch" and not isinstance(audio, torch.Tensor):
        if verbose:
            print("- Convert to Torch")
        audio = torch.Tensor(audio)
    elif return_format != "torch":
        if isinstance(audio, torch.Tensor):
            if verbose:
                print("- Convert from Torch to Numpy")
            audio = audio.numpy()
        elif isinstance(audio, list):
            if verbose:
                print("- Convert from list to Numpy")
            audio = np.array(audio, dtype=np.float32)
        if return_format == "bytes":
            if verbose:
                print("- Convert to bytes")
            audio = array_to_bytes(audio)

    return audio


def array_to_bytes(audio):
    return (audio * 32768).astype(np.int16).tobytes()


def save_audio(path, audio, sample_rate=16_000):
    """
    Save an audio signal into a wav file.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
        audio = audio.transpose()
    elif isinstance(audio, list):
        audio = np.array(audio, dtype=np.float32)
    sox.write(path, audio, sample_rate)


def get_audio_duration(path, verbose=False):
    """
    Return the duration of an audio file in seconds.
    """
    if os.path.isfile(path):
        try:
            info = sox.get_info(path)[0]
        except RuntimeError as e:
            raise RuntimeError(f"Error while reading {path}") from e
        return info.length / info.rate / info.channels
    return get_audio_total_duration(path, verbose=verbose)[1]


def get_audio_num_channels(path, verbose=False):
    """
    Return the number of channels in an audio file
    """
    assert os.path.isfile(path), f"File not found: {path}"
    info = sox.get_info(path)[0]
    return info.channels


def get_max_args():
    return int(run_command("getconf ARG_MAX"))


def get_audio_total_duration(files, max_args=1000, verbose=False):
    if max_args is None:
        max_args = get_max_args()
    assert max_args > 0
    total_duration = 0.0
    total_number = 0
    to_process = []
    for f in walk_files(files, ignore_extensions=[".json", ".csv", ".txt"], verbose=verbose, use_tqdm=verbose):
        to_process.append(f)
        if len(to_process) == max_args:
            nb, duration = _sox_duration(to_process)
            total_duration += duration
            total_number += nb
            to_process = []
    if len(to_process):
        nb, duration = _sox_duration(to_process)
        total_duration += duration
        total_number += nb
    return total_number, total_duration


def _sox_duration(files):
    # TODO: use soxi -D
    stdout = run_command(["soxi"] + files, False)
    last_break = stdout.rfind("\n")
    last_line = stdout[last_break:] if last_break >= 0 else ""
    if "Total Duration" in last_line:
        fields = last_line.split()
        duration = time2second(fields[-1])
        nb = int(fields[3])
        return nb, duration
    for line in stdout.split("\n"):
        if line.startswith("Duration"):
            duration = line.split()[2]
            return 1, time2second(duration)
    return 0, 0.0


def time2second(duration_str):
    h, m, s = map(float, duration_str.split(":"))
    seconds = h * 3600 + m * 60 + s
    return seconds


def mix_audios(files_in, file_out, method="stereo", ignore_existing=True):
    """
    Mix multiple audio files into a single file.

    Parameters
    ----------
    files_in: list of str
        list of input files
    file_out: str
        output file
    method: str
        'stereo': mix mono files in stereo
    ignore_existing: bool
        if True, do not overwrite the output file when it exists
    """
    assert file_out not in files_in, "Output file must not be included in input files"
    if ignore_existing and os.path.isfile(file_out):
        return
    for file in files_in:
        assert os.path.isfile(file), f"File not found: {file}"
        assert get_audio_num_channels(file) == 1, f"File {file} is not mono"
    if method == "stereo":
        ins = " -c 1 ".join(files_in)
        cmd = f"sox -M -c 1 {ins} {file_out}"
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    run_command(cmd)
