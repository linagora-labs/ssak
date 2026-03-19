import argparse
from tqdm import tqdm
from pathlib import Path
import json
from ssak.utils.nemo_dataset import NemoDataset

import pandas as pd
from IPython.display import display, clear_output
from IPython.display import Audio

import pandas as pd
import os

def strip_common_path(paths):
    """
    Remove the longest common directory prefix from a list of paths.
    """
    common = os.path.commonpath(paths)
    return {
        p: os.path.relpath(p, common)
        for p in paths
    }

def extract_audio_paths_from_item(json_row):
    """
    Extract audio file paths from conversation turns of type 'audio'.
    """
    audio_paths = []

    for turn in json_row.get("conversations", []):
        if turn.get("type") == "audio":
            value = turn.get("value")
            if value:
                audio_paths.append(value)

    return audio_paths

def display_data_interactive(merged_data, number_of_comparison=2, batch_size=5):
    keys = list(merged_data.keys())
    total_items = len(keys)
    current_index = 0

    def all_same(values):
        return len(set(str(v) for v in values)) == 1

    while current_index < total_items:
        end_index = min(current_index + batch_size, total_items)

        print("\n" + "=" * 100)
        print(f"Displaying items from {current_index + 1} to {end_index} out of {total_items}")
        print("=" * 100)

        for i in range(current_index, end_index):
            key = keys[i]
            items = merged_data[key]  # dict: manifest -> json_row

            print(f"\n{'─' * 100}")
            print(f"Item #{i + 1}")

            # ---- Short manifest names ---------------------------------------
            manifest_paths = list(items.keys())
            manifest_display = strip_common_path(manifest_paths)

            # ---- IDs ---------------------------------------------------------
            ids = [(m, item['id']) for m, item in items.items()]
            id_values = [v for _, v in ids]

            if all_same(id_values):
                print(f"ID: {id_values[0]}")
            else:
                print("IDs:")
                for m, v in ids:
                    print(f"  [{manifest_display[m]}] {v}")

            # ---- Align conversations by index --------------------------------
            max_len = max(len(item.get("conversations", [])) for item in items.values())

            for conv_idx in range(max_len):
                print(f"\n📋 Conversation #{conv_idx}")

                convs = []
                for manifest, item in items.items():
                    conv_list = item.get("conversations", [])
                    if conv_idx < len(conv_list):
                        convs.append((manifest, conv_list[conv_idx]))

                if not convs:
                    continue

                # Collect fields at this index
                fields = set()
                for _, conv in convs:
                    fields.update(conv.keys())

                fields.discard("type")

                for field in sorted(fields):
                    values = [
                        (m, conv.get(field))
                        for m, conv in convs
                        if field in conv
                    ]

                    if not values:
                        continue

                    only_values = [v for _, v in values]

                    # ---- Same across manifests --------------------------------
                    if all_same(only_values) and (len(only_values)>1 or number_of_comparison==1):
                        value = only_values[0]
                        print(f"  {field}:")
                        print(f"    {value}")

                    # ---- Different → show manifest ----------------------------
                    else:
                        print(f"  {field}:")
                        for m, v in values:
                            label = manifest_display[m]
                            print(f"    [{label}]")
                            print(f"      {v}")

        print("\n" + "=" * 100)

        current_index = end_index
        if current_index >= total_items:
            print(f"\n✓ Everything ({total_items}) has been displayed. ")
            break

        remaining = total_items - current_index
        user_input = input(
            f"\nPress Enter to display the next "
            f"{min(batch_size, remaining)} items, 'p' to play audio, '>X' to fast forward X data, or 'q' to quit: "
        ).strip().lower()

        if user_input == "q":
            print(f"\n✓ Stop displaying. {current_index}/{total_items} items were displayed.")
            break
        elif user_input == "p":
            audio_paths = extract_audio_paths_from_item(items)
            if not audio_paths:
                print("\n⚠ No audio turns found in this item.")
                continue

            for idx, path in enumerate(audio_paths):
                if os.path.exists(path):
                    print(f"\n▶ Playing audio turn #{idx}: {path}")
                    display(Audio(path))
                else:
                    print(f"\n⚠ File not found: {path}")
        elif user_input and user_input[0] == ">":
            if user_input[1:].isdigit():
                current_index += int(user_input[1:])
            elif all(i==">" for i in user_input):
                current_index += 100 * len(user_input)
            else:
                current_index += batch_size

# Utilisation
def process_path(paths):
    manifests = dict()
    for path in paths:
        dataset = NemoDataset()
        dataset.load(path)
        manifests[path] = dataset
    
    merged_data = dict()
    for manifest, data in manifests.items():
        for row in data:
            json_row = row.to_json()
            if not merged_data.get(json_row["id"]):
                 merged_data[json_row["id"]]= dict()
            merged_data[json_row["id"]][manifest] = json_row

    display_data_interactive(merged_data, number_of_comparison=len(paths), batch_size=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare fields of different manifests. You can also use to see the rows of a single manifest in a structured way."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="One or more manifests."
    )

    args = parser.parse_args()

    process_path(args.input_paths)
