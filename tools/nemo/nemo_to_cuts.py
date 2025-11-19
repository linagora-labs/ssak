import argparse
import json
from tqdm import tqdm
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet

def convert_manifest(input_path: str, output_cuts_path: str):
    if output_cuts_path is None:
        output_cuts_path = Path(input_path.replace("manifest", "cuts").replace(".jsonl", ".jsonl.gz"))
        print(f"Output cuts path was not provided, set to: {output_cuts_path}")
    elif not output_cuts_path.endswith(".jsonl.gz"):
        output_cuts_path = Path(output_cuts_path) / Path(Path(input_path).name.replace("manifest", "cuts").replace(".jsonl", ".jsonl.gz"))
        print(f"Output cuts path was a folder, set to: {output_cuts_path}")
    else:
        output_cuts_path = Path(output_cuts_path)
    output_cuts_path.parent.mkdir(parents=True, exist_ok=True)
    recordings = {}
    supervisions = []
    nemo_dataset = NemoDataset()
    data_type = nemo_dataset.load(input_path)
    for row in tqdm(nemo_dataset, desc="Processing lines"):
        audio = row.get_audio_turns()[0]
        audio_path = audio.audio_filepath
        recording_id = Path(audio_path).stem

        # Add recording if not already added
        if recording_id not in recordings:
            rec = Recording.from_file(audio_path, recording_id=recording_id)
            recordings[recording_id] = rec
        seg_id = row.id if row.id else f"{recording_id}_seg_{len(supervisions)}"
        # Create supervision segment
        seg = SupervisionSegment(
            id=row.id,
            recording_id=recording_id,
            start=audio.offset,
            duration=audio.duration,
            channel=0,
            text=row.answer,
            speaker=row.speaker,
            custom={"context": row.context} if row.context else None
        )
        supervisions.append(seg)

    print(f"Created {len(recordings)} recordings and {len(supervisions)} supervisions")

    # Build Lhotse sets
    recording_set = RecordingSet.from_recordings(recordings.values())
    supervision_set = SupervisionSet.from_segments(supervisions)

    # Combine into CutSet
    cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    cuts.to_file(output_cuts_path)

    print(f"Wrote Lhotse CutSet to: {output_cuts_path}")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Convert jsonl manifest to lhotse cuts."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help=""
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=""
    )
    args = parser.parse_args()
    convert_manifest(args.input_path, args.output_path)