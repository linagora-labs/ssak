#!/usr/bin/env python3

import os

import soundfile
from datasets import load_dataset
from tqdm import tqdm

from ssak.utils.kaldi_dataset import KaldiDataset


def write_dataset(
    huggingface_dataset,
    kaldi_dir="kaldi",
    wavs_path="wavs",
    existing_wavs_path=None,
    language=None,
    subset=None,
    trust_remote_code=False,
    set_name=None,
):
    if existing_wavs_path is None:
        os.makedirs(wavs_path, exist_ok=True)
    subset = language if language is not None else subset
    if set_name is not None and set_name in ["train", "test", "validation", "dev"]:
        data = load_dataset(huggingface_dataset, name=subset, split=set_name, trust_remote_code=trust_remote_code, streaming=True)
        write_split(data, set_name, kaldi_dir, wavs_path, existing_wavs_path)
    else:
        splits = load_dataset(huggingface_dataset, trust_remote_code=trust_remote_code, name=subset, streaming=True)
        for split in splits:
            data = load_dataset(huggingface_dataset, trust_remote_code=trust_remote_code, name=subset, split=split, streaming=True)
            write_split(data, split, os.path.join(kaldi_dir, set_name), wavs_path, existing_wavs_path)


def write_split(split_data, split_name, kaldi_dir, wavs_path, existing_wavs_path=None):
    dataset = KaldiDataset()
    for i, row in tqdm(enumerate(split_data), desc="Creating dataset"):
        id_ct = 0
        speaker = row["speaker_id"]
        audio_id = row["audio_id"]
        if existing_wavs_path is not None:
            audio_path = os.path.join(existing_wavs_path, row["audio"]["path"])
            if not os.path.isfile(audio_path):
                audio_path = os.path.join(existing_wavs_path, split_name, row["audio"]["path"])
                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Audio file {audio_path} not found")
            wav, sr = soundfile.read(audio_path)
            if round(len(wav) / sr, 4) != round(len(row["audio"]["array"]) / row["audio"]["sampling_rate"], 4):
                raise ValueError(f"Audio file {audio_path} does not match the expected length ({len(row['audio']['array'])} vs {len(wav)}) or sampling rate ({row['audio']['sampling_rate']} vs {sr})")
        else:
            audio_path = os.path.join(wavs_path, row["audio"]["path"])
            if not os.path.isfile(audio_path):
                soundfile.write(audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])
        for seg in row["segments"]:
            text = seg["transcript"].strip()
            if len(text) > 1:
                kaldi_row = {
                    "id": f"{audio_id}_{id_ct}",
                    "speaker": speaker,
                    "audio_id": audio_id,
                    "audio_path": audio_path,
                    "text": text,
                    "start": seg["start"],
                    "end": seg["end"],
                }
                dataset.append(kaldi_row)
                id_ct += 1
    dataset.save(kaldi_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert a dataset HuggingFace to Kaldi")
    parser.add_argument("--dataset", type=str, default="linagora/SUMM-RE", help="Dataset name")
    parser.add_argument("--kaldi_path", type=str, default="SUMM-RE", help="Path to new Kaldi directory")
    parser.add_argument(
        "--existing_wavs_path",
        type=str,
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/SUMM-RE_French_corpus/individual_files/audio",
        help="Path to store the audio files",
    )
    parser.add_argument(
        "--wavs_path",
        type=str,
        default="/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/SUMM-RE/wavs",
        help="Path to store the audio files",
    )
    parser.add_argument("--language", type=str, default=None, help="Language of the dataset if applicable")
    parser.add_argument("--subset", type=str, default=None, help="Language of the dataset if applicable")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust the remote code to run code locally. Default is False.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Name of the set to convert (train, test, validation). If None, all sets are converted.",
    )
    args = parser.parse_args()

    write_dataset(
        huggingface_dataset=args.dataset,
        kaldi_dir=args.kaldi_path,
        wavs_path=args.wavs_path,
        language=args.language,
        existing_wavs_path=args.existing_wavs_path,
        subset=args.subset,
        trust_remote_code=args.trust_remote_code,
        set_name=args.split,
    )
