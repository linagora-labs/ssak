import os

import numpy as np
import torch

from ssak.utils.audio import conform_audio, load_audio, save_audio

silero_vad_model = None
silero_get_speech_ts = None
pyannote_vad_pipeline = None


def get_vad_segments(
    audio,
    sample_rate=16_000,
    method="auditok",
    min_speech_duration=0.25,
    min_silence_duration=0.1,
    dilatation=0,
    output_sample=False,
    plot=False,
    verbose=False,
):
    """
    Get speech segments from audio file using Silero VAD
    parameters:
        audio: str or torch.Tensor
            path to audio file or audio data
        method: str
            method to use for VAD (silero, pyannote, auditok)
        sample_rate: int
            sample rate of audio data (in case it is not a file)
        min_speech_duration: float
            minimum speech duration (seconds)
        min_silence_duration: float
            minimum silence duration (seconds)
        dilatation: float
            dilatation of speech segments (seconds)
        output_sample: bool
            if True, return start and end in samples instead of seconds
        plot: bool
            if True, plot result
            if string, save plot to file
        verbose: bool
            if True, display information when loading models
    """
    global silero_vad_model, silero_get_speech_ts, pyannote_vad_pipeline

    method = method.lower()
    format = "torch"
    if method in ["silero"]:
        sample_rate_target = 16000
    elif method in ["pyannote", "auditok"]:
        sample_rate_target = None
    else:
        raise ValueError(f"Unknown VAD method: {method}")
    if method == "auditok":
        format = "array"

    if isinstance(audio, str):
        (audio, sample_rate) = load_audio(audio, sample_rate=None, return_format=format, verbose=verbose)
    audio = conform_audio(audio, sample_rate, sample_rate=sample_rate_target, return_format=format, verbose=verbose)

    if sample_rate_target is None:
        sample_rate_target = sample_rate

    if method == "silero":
        if silero_vad_model is None:
            if verbose:
                print("- Load Silero VAD model")
            import onnxruntime

            # Remove warning "Removing initializer 'XXX'. It is not used by any node and should be removed from the model."
            onnxruntime.set_default_logger_severity(3)
            silero_vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=True)
            silero_get_speech_ts = utils[0]

        # Cheap normalization of the amplitude
        audio = audio / max(0.1, audio.abs().max())

        segments = silero_get_speech_ts(
            audio,
            silero_vad_model,
            min_speech_duration_ms=round(min_speech_duration * 1000),
            min_silence_duration_ms=round(min_silence_duration * 1000),
            return_seconds=False,
        )

    elif method == "pyannote":
        if pyannote_vad_pipeline is None:
            if verbose:
                print("- Load pyannote")
            from pyannote.audio import Pipeline

            pyannote_vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=os.environ.get("HUGGINGFACE_TOKEN", None))

        pyannote_vad_pipeline.min_duration_on = min_speech_duration  # 0.05537587440407595
        pyannote_vad_pipeline.min_duration_off = min_silence_duration  # 0.09791355693027545
        # pyannote_vad_pipeline.onset = 0.8104268538848918
        # pyannote_vad_pipeline.offset = 0.4806866463041527

        pyannote_segments = pyannote_vad_pipeline({"waveform": audio.unsqueeze(0), "sample_rate": sample_rate_target})

        segments = []
        for speech_turn in pyannote_segments.get_timeline().support():
            segments.append({"start": speech_turn.start * sample_rate_target, "end": speech_turn.end * sample_rate_target})

    elif method == "auditok":
        import auditok

        data = (audio * 32767).astype(np.int16).tobytes()

        segments = auditok.split(
            data,
            sampling_rate=sample_rate_target,  # sampling frequency in Hz
            channels=1,  # number of channels
            sample_width=2,  # number of bytes per sample
            min_dur=min_speech_duration,  # minimum duration of a valid audio event in seconds
            max_dur=len(audio) / sample_rate_target,  # maximum duration of an event
            max_silence=min_silence_duration,  # maximum duration of tolerated continuous silence within an event
            energy_threshold=50,
            drop_trailing_silence=True,
        )

        if auditok.__version__ >= "0.3.0":

            def auditok_segment_to_dict(s):
                return {"start": s.start * sample_rate, "end": s.end * sample_rate}
        else:

            def auditok_segment_to_dict(s):
                return {"start": s._meta.start * sample_rate, "end": s._meta.end * sample_rate}

        segments = [auditok_segment_to_dict(s) for s in segments]

    if dilatation > 0:
        dilatation = round(dilatation * sample_rate_target)
        new_segments = []
        for seg in segments:
            new_seg = {"start": max(0, seg["start"] - dilatation), "end": min(len(audio), seg["end"] + dilatation)}
            if len(new_segments) > 0 and new_segments[-1]["end"] >= new_seg["start"]:
                new_segments[-1]["end"] = new_seg["end"]
            else:
                new_segments.append(new_seg)
        segments = new_segments

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        max_num_samples = 10000
        step = (audio.shape[-1] // max_num_samples) + 1
        times = [i * step / sample_rate_target for i in range((audio.shape[-1] - 1) // step + 1)]
        plt.plot(times, audio[::step])
        for s in segments:
            plt.axvspan(s["start"] / sample_rate_target, s["end"] / sample_rate_target, color="red", alpha=0.1)
        if isinstance(plot, str):
            plt.savefig(f"{plot}.VAD.jpg", bbox_inches="tight", pad_inches=0)
        else:
            plt.show()

    ratio = sample_rate / sample_rate_target if output_sample else 1 / sample_rate_target

    if ratio != 1.0:
        for seg in segments:
            seg["start"] *= ratio
            seg["end"] *= ratio
    if output_sample:
        for seg in segments:
            seg["start"] = round(seg["start"])
            seg["end"] = round(seg["end"])

    return segments


def remove_non_speech(
    audio,
    sample_rate=16_000,
    use_sample=False,
    path=None,
    plot=False,
    verbose=False,
    **kwargs,
):
    """
    Remove non-speech segments from audio (using Silero or Auditok, see get_vad_segments),
    glue the speech segments together and return the result along with
    a function to convert timestamps from the new audio to the original audio
    """
    if isinstance(audio, str):
        (audio, sample_rate) = load_audio(audio, sample_rate=None, return_format="torch", mono=False, verbose=verbose)

    segments = get_vad_segments(
        audio,
        sample_rate=sample_rate,
        output_sample=True,
        plot=plot,
        verbose=verbose,
        **kwargs,
    )

    segments = [(seg["start"], seg["end"]) for seg in segments]
    if len(segments) == 0:
        segments = [(0, audio.shape[-1])]
    if verbose:
        print(segments)

    if not isinstance(audio, torch.Tensor):
        audio_speech = np.concatenate([audio[..., s:e] for s, e in segments], axis=-1)
    else:
        audio_speech = torch.cat([audio[..., s:e] for s, e in segments], dim=-1)

    if len(audio.shape) == 1:
        audio_speech = audio_speech.reshape(-1)

    if path:
        if verbose:
            print(f"Save audio to {path}")
        save_audio(path, audio_speech, sample_rate)

    if not use_sample:
        segments = [(float(s) / sample_rate, float(e) / sample_rate) for s, e in segments]
        if verbose:
            print(segments)

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        max_num_samples = 10000
        audio_speech_mono = audio_speech.mean(dim=0) if len(audio_speech.shape) > 1 else audio_speech
        step = (audio_speech_mono.shape[-1] // max_num_samples) + 1
        times = [i * step / sample_rate for i in range((audio_speech_mono.shape[-1] - 1) // step + 1)]
        plt.plot(times, audio_speech_mono[::step])
        if isinstance(plot, str):
            plt.savefig(f"{plot}.speech.jpg", bbox_inches="tight", pad_inches=0)
        else:
            plt.show()

    # if not isinstance(audio, torch.Tensor):
    #     audio_speech = audio_speech.numpy()

    return audio_speech, lambda t, t2=None: convert_timestamps(segments, t, t2)


def convert_timestamps(segments, t, t2=None):
    """
    Convert timestamp from audio without non-speech segments to original audio (with non-speech segments)

    parameters:
        segments: list of tuple (start, end) corresponding to non-speech segments in original audio
        t: timestamp to convert
        t2: second timestamp to convert (optional), when the two timestamps should be in the same segment
    """
    assert len(segments)
    ioffset = 0  # Input offset
    ooffset = 0  # Output offset
    ipreviousend = 0
    result = []
    for istart, iend in segments:
        ostart = ooffset
        oend = ostart + (iend - istart)
        ooffset = oend
        ioffset += istart - ipreviousend
        ipreviousend = iend
        t_in = t <= oend
        t2_in = t_in if t2 is None else t2 <= oend
        if t_in or t2_in:
            result.append([max(istart, min(iend, ioffset + t)), max(istart, min(iend, ioffset + t2)) if t2 is not None else None])
            if t_in and t2_in:
                break
    if not len(result):
        result.append([ioffset + t, ioffset + t2 if t2 is not None else None])

    if len(result) > 1:
        # Minimize difference between durations
        result = sorted(result, key=lambda x: abs(abs(t2 - t) - abs(x[1] - x[0])))
    result = result[0]
    if t2 is None:
        result = result[0]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove non-speech segments from audio file and show the result",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_file", nargs="+", help="path to audio file(s)")
    parser.add_argument("--method", default="auditok", choices=["silero", "pyannote", "auditok"], help="VAD method")
    parser.add_argument("--min_speech_duration", type=float, default=0.25, help="minimum speech duration (seconds)")
    parser.add_argument("--min_silence_duration", type=float, default=0.1, help="minimum silence duration (seconds)")
    parser.add_argument("--dilatation", type=float, default=0, help="dilatation of speech segments (seconds)")
    parser.add_argument("--save_output", action="store_true", help="save audio without silence (beside the file)")
    parser.add_argument("--disable_plot", action="store_true", help="do not plot results")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    args = parser.parse_args()

    kwargs = dict(
        method=args.method,
        min_speech_duration=args.min_speech_duration,
        min_silence_duration=args.min_silence_duration,
        dilatation=args.dilatation,
        plot=not args.disable_plot,
        verbose=args.verbose,
    )

    for audio_file in args.audio_file:
        if args.verbose:
            print(f"Processing {audio_file}...")

        audio, func = remove_non_speech(
            audio_file,
            path=f"{audio_file}.speech_{args.method}.mp3" if args.save_output else None,
            **kwargs,
        )
