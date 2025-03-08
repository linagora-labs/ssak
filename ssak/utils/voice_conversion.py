import argparse
import glob
import os
import random
import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import so_vits_svc_fork.f0
import soundfile as sf
import torch
from so_vits_svc_fork.inference.core import Svc, split_silence
from tqdm import tqdm

from ssak.utils.audio import load_audio
from ssak.utils.dataset import kaldi_folder_to_dataset

# Ignore all warnings
warnings.simplefilter("ignore")


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Audio inference using SVC model")

    # paths
    parser.add_argument(
        "kaldi_input",
        type=str,
        help="Input Kaldi folder",
    )
    parser.add_argument(
        "model_base_path",
        type=str,
        help="Path to the SVC models. To train so-vits-svc you can visit their GitHub: https://github.com/voicepaw/so-vits-svc-fork",
    )
    parser.add_argument("-ok", "--kaldi_output", type=str, default=None, help="Output kaldi folder")
    parser.add_argument(
        "-o",
        "--audio_output_path",
        type=str,
        default=None,
        help="Output path or directory for the processed audio files",
    )
    parser.add_argument(
        "-ms",
        "--max_spk",
        type=str,
        default="1",
        help="Maximum speakers to use",
    )
    parser.add_argument(
        "-s",
        "--speaker",
        type=str,
        default=None,
        help="Speaker to use",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        type=int,
        default=0,
        help="Transpose factor for pitch shifting (default: 0)",
    )
    parser.add_argument(
        "-a",
        "--auto_predict_f0",
        action="store_true",
        help="Automatically predict F0 (default: False)",
    )
    parser.add_argument(
        "-cl",
        "--cluster_infer_ratio",
        type=float,
        default=0,
        help="Ratio of clusters to infer from the cluster model (default: 0)",
    )
    parser.add_argument(
        "-ns",
        "--noise_scale",
        type=float,
        default=0.4,
        help="Noise scale for wave synthesis (default: 0.4)",
    )
    parser.add_argument(
        "-f0",
        "--f0_method",
        type=str,
        choices=["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
        default="crepe",
        help="F0 estimation method (default: crepe)",
    )
    # slice config
    parser.add_argument(
        "--db_thresh",
        type=int,
        default=-40,
        help="Decibel threshold for silence detection (default: -40)",
    )
    parser.add_argument(
        "--pad_seconds",
        type=float,
        default=0.5,
        help="Padding duration in seconds for slicing (default: 0.5)",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=0.5,
        help="Chunk duration in seconds for slicing (default: 0.5)",
    )
    parser.add_argument(
        "-ab",
        "--absolute_thresh",
        action="store_true",
        help="Use absolute threshold for silence detection (default: False)",
    )
    parser.add_argument(
        "--voice_change_mode",
        choices=["per_chunk", "per_segment"],
        default="per_chunk",
        help="Mode of voice change (per_chunk or per_segment)",
    )
    parser.add_argument("--max_chunk_seconds", type=int, default=40, help="Mode of voice change (per_chunk or per_segment)")

    # device
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)",
    )

    return parser


def get_initials(names: list) -> str:
    initials = {}
    result = []

    for name in names:
        if "_" in name:
            parts = name.split("_")
            initial = parts[0][0] + parts[1][0] if len(parts) > 1 else parts[0][0]
        else:
            initial = name[:2]

        # Check for duplicate initials
        if initial in initials:
            # Append the next character to resolve the conflict
            current_length = len(initials[initial])
            initials[initial].append(name[current_length])
            # Update the result for all conflicting initials
            for idx in initials[initial]:
                result[idx] = result[idx] + name[current_length]
        else:
            initials[initial] = [len(result)]
            result.append(initial)

    return "_".join(result)


def read_and_generate_segment_dict(kaldi_dir):
    _, dataframe = kaldi_folder_to_dataset(kaldi_dir, return_format="pandas")
    segments = {}
    segment_dict = {}
    for _, row in dataframe.iterrows():
        audio_id = os.path.basename(row["path"]).split(".")[0]
        seg_id = row["ID"]
        text = row["text"]
        start_time = row["start"]
        end_time = row["end"]
        audio_path = row["path"]
        segments.setdefault(audio_id, []).append({"seg_id": seg_id, "text": text, "start": start_time, "end": end_time, "path": audio_path})
    for audio_id, segment_list in segments.items():
        segment_dict[audio_id] = {}
        for segment in segment_list:
            seg_id = segment["seg_id"]
            segment_dict[audio_id][seg_id] = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "wave_path": segment["path"],
            }
    return segment_dict


def find_speakers(path):
    dict_of_spk = set()
    if not os.path.exists(path):
        print("The provided path does not exist.")
        return dict_of_spk
    for _, dirs, _ in os.walk(path):
        for dir_name in dirs:
            dict_of_spk.add(dir_name)
    return dict_of_spk


def convert_to_int(s):
    if s.isdigit():
        return int(s)
    else:
        return str(s)


def _select_speakers(speakers: list[str], max_spk: Union[str, int]) -> list[str]:
    if not speakers:
        print("The speaker list is empty!")
        return []
    if isinstance(max_spk, str) and max_spk.lower() == "all":
        return speakers
    if isinstance(max_spk, int):
        if max_spk == 1:
            return random.sample(speakers, 1)
        if max_spk >= len(speakers):
            return speakers
        return random.sample(speakers, max_spk)

    print(f"Invalid max_spk value: {max_spk}")
    return []


def _get_speakers(speaker: str, model_base_path: Path) -> list[str]:
    if speaker is None:
        speakers = find_speakers(model_base_path)
    elif "," in speaker:
        speakers = speaker.split(",")
        speakers = [spk.strip() for spk in speakers]
    else:
        speakers = [speaker]
    return speakers


def _load_svc_models(speakers: list, model_base_path: Path, device: torch.device) -> dict:
    models = {}
    for spk in speakers:
        speaker_model_path = Path(model_base_path) / spk
        file_paths = glob.glob(str(speaker_model_path / "G_*.pth"))
        kmeans_file = glob.glob(str(speaker_model_path / "kmeans*.pt"))

        if not file_paths:
            print(f"No model files found for speaker: {spk}")
            continue

        latest_model_path = Path(max(file_paths, key=os.path.getmtime))
        latest_kmeans_path = Path(max(kmeans_file, key=os.path.getmtime)) if kmeans_file else None
        config_path = speaker_model_path / "config.json"
        models[spk] = {
            "model_path": latest_model_path,
            "config_path": config_path,
            "cluster_model_path": latest_kmeans_path,
        }

    svc_models = {}
    for spk, paths in models.items():
        svc_model = Svc(
            net_g_path=paths["model_path"].as_posix(),
            config_path=paths["config_path"].as_posix(),
            cluster_model_path=paths["cluster_model_path"].as_posix() if paths["cluster_model_path"] else None,
            device=device,
        )
        svc_models[spk] = svc_model
    return svc_models


def _convert_voice(
    *,
    input_path: Union[Path, str],
    model_base_path: Union[Path, str],
    kaldi_output: Union[Path, str],
    audio_output_path: Union[Path, str],
    max_spk: Union[str, int] = 1,
    speaker: str,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "crepe",
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
    device: Union[str, torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    voice_change_mode: Literal["per_chunk", "per_segment"] = "per_chunk",
    max_chunk_seconds: int = 0,
):
    try:
        if isinstance(input_path, str):
            input_path = Path(input_path)
        if isinstance(audio_output_path, str):
            audio_output_path = Path(audio_output_path)
        if isinstance(kaldi_output, str):
            kaldi_output = Path(kaldi_output)

        speakers = _get_speakers(speaker, model_base_path)
        selected_speaker_models = _select_speakers(speakers, max_spk)
        spkname = get_initials(selected_speaker_models)

        print(f"\nChosen SPK :{selected_speaker_models}\n")
        svc_models = _load_svc_models(selected_speaker_models, model_base_path, device)

        if kaldi_output is None:
            kf_basename = input_path.stem
            kaldi_output = input_path.parent / f"{kf_basename}_augmented_{spkname}"

        os.makedirs(kaldi_output, exist_ok=True)

        with open(kaldi_output / "text", "w", encoding="utf-8") as text, open(kaldi_output / "utt2spk", "w", encoding="utf-8") as utt2spk, open(kaldi_output / "spk2utt", "w", encoding="utf-8") as spk2utt, open(
            kaldi_output / "wav.scp", "w", encoding="utf-8"
        ) as wav_scp, open(kaldi_output / "segments", "w", encoding="utf-8") as segments, open(kaldi_output / "utt2dur", "w", encoding="utf-8") as utt2dur:
            wave_segments = read_and_generate_segment_dict(input_path.as_posix())

            for wave_id, seg_info_dict in tqdm(wave_segments.items(), desc="Processing Waves", unit="wave"):
                concat_audio = None
                audio_path = None
                waveform = None
                output_file_path = None
                for segment_id, seg_info in tqdm(seg_info_dict.items(), desc="Processing Segments", unit="segment") if voice_change_mode == "per_segment" else seg_info_dict.items():
                    random_spk = random.choice(list(svc_models.keys()))
                    random_svc_model = svc_models[random_spk]
                    try:
                        audio_path = seg_info["wave_path"]

                        if audio_path is None or not os.path.exists(audio_path):
                            raise ValueError(f"Invalid or missing audio path for segment {segment_id}")

                        seg_id_with_prefix = f"{spkname}_augmented_{segment_id}"
                        wave_id_sp = f"{spkname}_{wave_id}"
                        text_segment = seg_info["text"]
                        start_time = seg_info["start"]
                        end_time = seg_info["end"]
                        duration = end_time - start_time

                        text.write(f"{seg_id_with_prefix} {text_segment}\n")
                        utt2spk.write(f"{seg_id_with_prefix} {seg_id_with_prefix}\n")
                        spk2utt.write(f"{seg_id_with_prefix} {seg_id_with_prefix}\n")
                        segments.write(f"{seg_id_with_prefix} {wave_id_sp} {start_time} {end_time}\n")
                        utt2dur.write(f"{seg_id_with_prefix} {duration}\n")

                        if audio_output_path is None:
                            audio_file_path = Path(audio_path)
                            audio_folder = audio_file_path.parent
                            audio_output_path = audio_folder.with_name(f"audio_augmented_{spkname}")
                        audio_output_path.mkdir(parents=True, exist_ok=True)

                        output_file_path = audio_output_path / f"{wave_id_sp}.wav"

                        if voice_change_mode == "per_segment":
                            if output_file_path.exists():
                                print(f"Skipping already processed file: {output_file_path}")
                                continue

                            waveform = load_audio(audio_path, start=start_time, end=end_time, sample_rate=random_svc_model.target_sample)
                            audio = random_svc_model.infer_silence(
                                waveform,
                                speaker=random_spk,
                                transpose=transpose,
                                auto_predict_f0=auto_predict_f0,
                                cluster_infer_ratio=cluster_infer_ratio,
                                noise_scale=noise_scale,
                                f0_method=f0_method,
                                db_thresh=db_thresh,
                                pad_seconds=pad_seconds,
                                chunk_seconds=chunk_seconds,
                                absolute_thresh=absolute_thresh,
                                max_chunk_seconds=duration,
                            )

                            if concat_audio is None:
                                concat_audio = audio
                            else:
                                concat_audio = np.concatenate((concat_audio, audio))

                    except Exception as e:
                        print(f"Failed to process {segment_id}")
                        print(e)
                        continue

                if voice_change_mode == "per_chunk":
                    if output_file_path.exists():
                        print(f"Skipping already processed file: {output_file_path}")
                        continue

                    waveform = load_audio(audio_path, sample_rate=44100)
                    if len(waveform.shape) > 1:
                        waveform = waveform.T  # Swap axes if the audio is stereo.

                    max_chunk_seconds = max_chunk_seconds if max_chunk_seconds > 0 else 40

                    chunk_length_min = int(min(44100 / so_vits_svc_fork.f0.f0_min * 20 + 1, chunk_seconds * 44100)) // 2

                    chunks = split_silence(
                        waveform,
                        top_db=-db_thresh,
                        frame_length=chunk_length_min * 2,
                        hop_length=chunk_length_min,
                        ref=1 if absolute_thresh else np.max,
                        max_chunk_length=int(max_chunk_seconds * 44100),
                    )

                    for chunk in tqdm(chunks, desc="Processing Chunks", unit="chunk"):
                        random_spk = random.choice(list(svc_models.keys()))
                        random_svc_model = svc_models[random_spk]

                        if not chunk.is_speech:  # Assuming chunk has an is_speech attribute
                            audio_chunk_infer = np.zeros_like(chunk.audio)
                        else:
                            pad_len = int(44100 * pad_seconds)
                            audio_chunk_pad = np.concatenate(
                                [
                                    np.zeros([pad_len], dtype=np.float32),
                                    chunk.audio,
                                    np.zeros([pad_len], dtype=np.float32),
                                ]
                            )
                            audio_chunk_pad_infer_tensor, _ = random_svc_model.infer(
                                random_spk,
                                transpose,
                                audio_chunk_pad,
                                cluster_infer_ratio=cluster_infer_ratio,
                                auto_predict_f0=auto_predict_f0,
                                noise_scale=noise_scale,
                                f0_method=f0_method,
                            )
                            audio_chunk_pad_infer = audio_chunk_pad_infer_tensor.cpu().numpy()
                            cut_len_2 = (len(audio_chunk_pad_infer) - len(chunk.audio)) // 2
                            audio_chunk_infer = audio_chunk_pad_infer[cut_len_2 : cut_len_2 + len(chunk.audio)]
                            torch.cuda.empty_cache()

                        if concat_audio is None:
                            concat_audio = audio_chunk_infer
                        else:
                            concat_audio = np.concatenate([concat_audio, audio_chunk_infer])
                    concat_audio = concat_audio[: waveform.shape[0]]

                if concat_audio is not None:
                    wav_scp.write(f"{spkname}_{wave_id} sox {output_file_path} -t wav -r 16000 -b 16 -c 1 - |\n")
                    sf.write(output_file_path, concat_audio, random_svc_model.target_sample)

    except Exception as ex:
        print(f"Exception occurred: {ex}")

    finally:
        if "random_svc_model" in locals():
            del random_svc_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    max_spk = convert_to_int(args.max_spk)

    _convert_voice(
        input_path=args.kaldi_input,
        model_base_path=args.model_base_path,
        audio_output_path=args.audio_output_path,
        kaldi_output=args.kaldi_output,
        max_spk=max_spk,
        speaker=args.speaker,
        transpose=args.transpose,
        auto_predict_f0=args.auto_predict_f0,
        cluster_infer_ratio=args.cluster_infer_ratio,
        noise_scale=args.noise_scale,
        f0_method=args.f0_method,
        db_thresh=args.db_thresh,
        pad_seconds=args.pad_seconds,
        chunk_seconds=args.chunk_seconds,
        absolute_thresh=args.absolute_thresh,
        device=args.device,
        voice_change_mode=args.voice_change_mode,
        max_chunk_seconds=args.max_chunk_seconds,
    )
