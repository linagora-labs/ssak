import argparse
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet

def convert_manifest(input_path: str, output_cuts_path: str):
    """Fast conversion that processes data in streaming fashion without loading everything into memory first."""
    if output_cuts_path is None:
        output_cuts_path = Path(input_path.replace("manifest", "cuts").replace(".jsonl", ".jsonl.gz"))
        print(f"Output cuts path was not provided, set to: {output_cuts_path}")
    elif not output_cuts_path.endswith(".jsonl.gz"):
        output_cuts_path = Path(output_cuts_path) / Path(Path(input_path).name.replace("manifest", "cuts").replace(".jsonl", ".jsonl.gz"))
        print(f"Output cuts path was a folder, set to: {output_cuts_path}")
    else:
        output_cuts_path = Path(output_cuts_path)
    
    output_cuts_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use sets and dictionaries for fast lookups
    seen_recordings = set()
    recordings_data = {}
    supervisions_data = []
    
    # Get total lines for progress bar
    with open(input_path, 'r', encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Processing {total_lines} lines...")
    
    # Stream process the file
    with open(input_path, 'r', encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing lines")):
            json_row = json.loads(line)
            
            # Determine data type from first line
            if line_num == 0:
                data_type = "multiturn" if "conversations" in json_row else "asr"
                print(f"Detected data type: {data_type}")
            
            # Parse row based on data type
            if data_type == "asr":
                audio_path = json_row["audio_filepath"]
                text = json_row["text"]
                duration = json_row["duration"]
                offset = json_row.get("offset", 0.0)
                row_id = json_row.get("id", json_row.get("utt_id"))
                speaker = json_row.get("speaker")
                context = json_row.get("context")
            else:  # multiturn
                conversations = json_row["conversations"]
                # Find audio and text turns
                audio_turn = None
                text_turn = None
                context_turn = None
                
                for turn in conversations:
                    if turn.get("type") == "audio":
                        audio_turn = turn
                    elif turn.get("type") == "text":
                        if turn.get("from") == "User" and text_turn is None:
                            context_turn = turn
                        elif turn.get("from") == "Assistant" or text_turn is None:
                            text_turn = turn
                
                if not audio_turn or not text_turn:
                    continue
                
                audio_path = audio_turn["value"]
                text = text_turn["value"]
                duration = audio_turn["duration"]
                offset = audio_turn.get("offset", 0.0)
                row_id = json_row.get("id", json_row.get("utt_id"))
                speaker = json_row.get("speaker")
                context = context_turn["value"] if context_turn else None
            
            recording_id = Path(audio_path).stem
            
            # Store recording data (defer creation)
            if recording_id not in seen_recordings:
                recordings_data[recording_id] = audio_path
                seen_recordings.add(recording_id)
            
            # Create supervision data
            seg_id = row_id if row_id else f"{recording_id}_seg_{len(supervisions_data)}"
            supervision_data = {
                "id": seg_id,
                "recording_id": recording_id,
                "start": offset,
                "duration": duration,
                "channel": 0,
                "text": text,
                "speaker": speaker,
                "custom": {"context": context} if context else None
            }
            supervisions_data.append(supervision_data)
    
    print(f"Collected {len(recordings_data)} unique recordings and {len(supervisions_data)} supervisions")
    print("Creating Lhotse objects...")
    
    # Batch create recordings
    recordings = []
    for recording_id, audio_path in tqdm(recordings_data.items(), desc="Creating recordings"):
        rec = Recording.from_file(audio_path, recording_id=recording_id)
        recordings.append(rec)
    
    # Batch create supervisions
    supervisions = []
    for sup_data in tqdm(supervisions_data, desc="Creating supervisions"):
        seg = SupervisionSegment(**sup_data)
        supervisions.append(seg)
    
    print("Building Lhotse sets...")
    # Build Lhotse sets
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    
    print("Creating CutSet...")
    # Combine into CutSet
    cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    
    print(f"Writing to {output_cuts_path}...")
    cuts.to_file(output_cuts_path)
    
    print(f"Successfully wrote Lhotse CutSet to: {output_cuts_path}")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Convert jsonl manifest to lhotse cuts."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Input manifest"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Folder or file (.jsonl.gz)"
    )
    args = parser.parse_args()
    convert_manifest(args.input_path, args.output_path)