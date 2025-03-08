import argparse
import os

import torch
from tqdm import tqdm

from ssak.utils.audio import load_audio
from ssak.utils.dataset import kaldi_folder_to_dataset
from ssak.utils.gender import predict_gender
from ssak.utils.kaldi import check_kaldi_dir

# Ignore all warnings
# import warnings
# warnings.simplefilter("ignore")


def main(args):
    # Check output file does not exist
    kaldi_spk2g_file = os.path.join(args.kaldi_dir, "spk2gender")
    if os.path.exists(kaldi_spk2g_file):
        print("WARNING!! File already exists!")
        return

    # Read speakers
    utt2spk_file = os.path.join(args.kaldi_dir, "utt2spk")
    if not os.path.exists(utt2spk_file):
        raise RuntimeError("ERROR!! File not found: utt2spk")
    utt2spk = {}
    with open(utt2spk_file, encoding="utf-8") as utt2spk_f:
        for line in utt2spk_f:
            fields = line.strip().split()
            assert len(fields) == 2
            utt, spk = fields
            utt2spk[utt] = spk

    spk2gender = {}
    stats, dataset = kaldi_folder_to_dataset(args.kaldi_dir, online=True)
    for sample in tqdm(dataset, desc="Processing audio segments", total=stats["samples"]):
        utt = sample["ID"]
        assert utt in utt2spk, f"Speaker not found for utterance {utt}"
        spk = utt2spk[utt]
        audio_path = sample["path"]
        start = sample["start"]
        end = sample["end"]
        waveform = load_audio(audio_path, start=start, end=end)
        gender = predict_gender(waveform, device=args.device, model=args.model_path, output_type="scores")
        # Weight scores by (clipped) duration of the utterance
        weight = min(10, waveform.size / 16_000)
        m_score, f_score = gender["m"], gender["f"]
        spk2gender[spk] = spk2gender.get(spk, []) + [(m_score * weight, f_score * weight)]

    with open(kaldi_spk2g_file, "w", encoding="utf-8") as f:
        for spk, scores in spk2gender.items():
            m_scores, f_scores = zip(*scores)
            m_score = sum(m_scores)
            f_score = sum(f_scores)
            gender = "m" if m_score > f_score else "f"
            f.write(f"{spk} {gender}\n")

    check_kaldi_dir(args.kaldi_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender prediction for audio segments.")
    parser.add_argument("kaldi_dir", type=str, help="Path to the directory containing Kaldi data.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="m3hrdadfi/hubert-base-persian-speech-gender-recognition",
        help="Path to the pretrained model or its name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cpu or cuda).",
    )
    args = parser.parse_args()

    main(args)
