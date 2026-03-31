
import json
import requests
import librosa
import random
import shutil
import numpy as np
import soundfile as sf
from datasets import load_dataset
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

random.seed(42)
np.random.seed(42)

prompts = [
    "Describe this music in a few sentences.",
    "Analyze the music and describe what you hear",
    "Tell me about the music you hear.",
    "Summarize the music : genre, strings, mood, where can it be played, quality etc...",
    "Please describe the music",
    "Give me a report on the music played",
    "What are the characteristics of this music?",
    "What can you say about this music recording?",
    "Write down a complete breakdown of the music",
    "How would you describe this musical excerpt?",
    "Write a detailed description of the music, including the recording quality, musical instruments, and any other specific features.",
]

if __name__ == "__main__":
    # Path to save audios

    OUTPUT_FOLDER = Path("/data-server/datasets/audio/raw/music/MusicCaps")
    AUDIO_FOLDER = OUTPUT_FOLDER / "audios"
    AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
    
    nemo_no_context_folder = Path("/data-server/datasets/audio/nemo/music/music-captioning/en/nocontext/MusicCaps")
    nemo_context_folder = Path("/data-server/datasets/audio/nemo/music/music-captioning/en/context/MusicCaps")

    train_dataset = NemoDataset()
    test_dataset = NemoDataset()
    
    # if not Path(OUTPUT_FOLDER / "test.jsonl").exists():
    dataset = load_dataset("CLAPv2/MusicCaps", streaming=True)
    for idx, sample in enumerate(dataset["train"]):
        audio_data = sample["audio"]["array"]     # the audio array
        sample_rate = sample["audio"]["sampling_rate"]

        caption = sample["caption"]

        audio_path = AUDIO_FOLDER / sample['index']
        if not audio_path.exists():
            audio_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sf.write(audio_path, audio_resampled, 16000, subtype="PCM_16")

        nemo_row = NemoDatasetRow(
            id=sample['index'],
            turns=[
                NemoTurn(role="User", value=str(audio_path), turn_type="audio", duration=round(sample["audio_len"],3)),
                NemoTurn(role="Assistant", value=caption, turn_type="text")
            ],
            custom_metadata={"aspect_list": sample["aspect_list"]}
        )
        if isinstance(sample["is_audioset_eval"], str):
            if sample["is_audioset_eval"].strip() not in ["True", "False"]:
                print(f"Error {sample}")
            sample["is_audioset_eval"] = sample["is_audioset_eval"].strip() == "True"
        if sample["is_audioset_eval"]==True:
            test_dataset.append(nemo_row)
        else:
            train_dataset.append(nemo_row)

        if idx % 100 == 0:
            print(f"Processed {idx} samples...")
    train_dataset.save(OUTPUT_FOLDER / "train.jsonl")
    test_dataset.save(OUTPUT_FOLDER / "test.jsonl")
    shutil.copyfile(OUTPUT_FOLDER / "train.jsonl", nemo_no_context_folder / "train.jsonl")
    shutil.copyfile(OUTPUT_FOLDER / "test.jsonl", nemo_no_context_folder / "test.jsonl")
    train_dataset.set_context_if_none(prompts)
    train_dataset.save(nemo_context_folder / "train.jsonl")
    test_dataset.set_context_if_none(prompts)
    test_dataset.save(nemo_context_folder / "test.jsonl")
    print("Done")
