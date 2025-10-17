import os

import numpy as np
from datasets import Audio, Dataset
from huggingface_hub import HfApi, HfFolder
from linastt.utils.dataset import kaldi_folder_to_dataset


def gen_hf_data(dataset):
    audio_dict = {}  # Dictionary to store unique audio paths and their metadata
    ids = []  # Stores IDs
    all_segments = []  # Stores all segments
    transcript = ""
    current_segments = []
    id_p = None  # Previous ID

    for data in dataset:
        # Extract the base ID (remove any segment suffix)
        id = data["ID"] if "-seg" not in data["ID"] else data["ID"].split("-seg")[0]

        if id != id_p:
            if id_p is not None:
                # Append the previous id's data
                all_segments.append(current_segments)
                if audio_path not in audio_dict:
                    audio_dict[audio_path] = {"segments": current_segments, "transcript": transcript}
                else:
                    audio_dict[audio_path]["segments"].extend(current_segments)
                    audio_dict[audio_path]["transcript"] += " " + transcript

            ids.append(id)
            audio_path = data["path"]

            # Initialize new transcript and segments for the new id
            transcript = data["text_norm"]
            current_segments = [
                {
                    "start": np.float64(data["start"]),
                    "end": np.float64(data["end"]),
                    "transcript": str(data["text"]),
                    "transcript_raw": str(data["text_norm"]),
                }
            ]
        else:
            # Append to existing transcript and segments
            transcript += " " + data["text_norm"]
            current_segments.append(
                {
                    "start": np.float64(data["start"]),
                    "end": np.float64(data["end"]),
                    "transcript": str(data["text"]),
                    "transcript_raw": str(data["text_norm"]),
                }
            )

        id_p = id

    # Append the last id's data
    if id_p is not None:
        all_segments.append(current_segments)
        if audio_path not in audio_dict:
            audio_dict[audio_path] = {"segments": current_segments, "transcript": transcript}
        else:
            audio_dict[audio_path]["segments"].extend(current_segments)
            audio_dict[audio_path]["transcript"] += " " + transcript

    # Convert dictionary to lists for Dataset
    audio_paths = list(audio_dict.keys())
    segments = [audio_dict[path]["segments"] for path in audio_paths]
    transcripts = [audio_dict[path]["transcript"] for path in audio_paths]

    dict_data = {
        "audio_id": ids,
        "audio": audio_paths,
        "segments": segments,
        "transcript": transcripts,
    }

    # Create and return the dataset
    dict_dataset = Dataset.from_dict(dict_data).cast_column("audio", Audio())

    return dict_dataset


if __name__ == "__main__":
    import sys

    token_hf = sys.argv[1]
    path_data = sys.argv[2]
    split = sys.argv[3]

    repo_name_base = "linagora/linto-dataset-audio-ar-tn-0.1"

    # Initialize Hugging Face API
    api = HfApi()
    HfFolder.save_token(token_hf)

    # Load data
    data_name = os.path.basename(path_data)
    _, data = kaldi_folder_to_dataset(path_data)

    # Generate dataset
    dataset = gen_hf_data(data)

    # Push to Hugging Face Hub
    dataset.push_to_hub(repo_name_base, data_dir=f"data/{data_name}/{split}", split=split)
