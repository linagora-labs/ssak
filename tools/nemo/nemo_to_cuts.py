import argparse
from pathlib import Path

from ssak.utils.nemo_dataset import NemoDataset
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
    
    if output_cuts_path.exists():
        print(f"Output cuts path already exists: {output_cuts_path}")
        return
    
    output_cuts_path.parent.mkdir(parents=True, exist_ok=True)
    
    recordings = {}
    supervisions = []
    
    nemo_dataset = NemoDataset()
    
    for row in nemo_dataset.stream(input_path):
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
            id=seg_id,
            recording_id=recording_id,
            start=audio.offset,
            duration=audio.duration,
            channel=0,
            text=row.answer,
            speaker=row.speaker,
            custom={"context": row.context} if row.context else None
        )
        supervisions.append(seg)
    
    recording_set = RecordingSet.from_recordings(recordings.values())
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