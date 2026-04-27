import argparse
from pathlib import Path

from ssak.utils.nemo_dataset import NemoDataset
from lhotse import Recording, SupervisionSegment, MonoCut, CutSet


def convert_manifest(input_path: str, output_cuts_path: str, force: bool = False):
    """Fast conversion that processes data in streaming fashion without loading everything into memory first.

    Builds one MonoCut per row with a single supervision, so datasets like Clotho-AQA
    (N Q&A pairs sharing the same audio file) don't collapse into a single cut
    with N overlapping supervisions.
    """
    if output_cuts_path is None:
        output_cuts_path = Path(input_path.replace("manifest", "cuts").replace(".jsonl", ".jsonl.gz"))
        print(f"Output cuts path was not provided, set to: {output_cuts_path}")
    elif not output_cuts_path.endswith(".jsonl.gz"):
        output_cuts_path = Path(output_cuts_path) / Path(Path(input_path).name.replace("manifest", "cuts").replace(".jsonl", ".jsonl.gz"))
        print(f"Output cuts path was a folder, set to: {output_cuts_path}")
    else:
        output_cuts_path = Path(output_cuts_path)

    if output_cuts_path.exists() and not force:
        print(f"Output cuts path already exists (use --force to overwrite): {output_cuts_path}")
        return

    output_cuts_path.parent.mkdir(parents=True, exist_ok=True)

    recordings_cache = {}
    cuts = []

    nemo_dataset = NemoDataset()

    for row in nemo_dataset.stream(input_path):
        audio = row.get_audio_turns()[0]
        audio_path = audio.audio_filepath
        recording_id = Path(audio_path).stem

        if recording_id not in recordings_cache:
            recordings_cache[recording_id] = Recording.from_file(audio_path, recording_id=recording_id)
        rec = recordings_cache[recording_id]

        seg_id = row.id if row.id else f"{recording_id}_seg_{len(cuts)}"

        seg = SupervisionSegment(
            id=seg_id,
            recording_id=recording_id,
            start=audio.offset,
            duration=audio.duration,
            channel=0,
            text=row.text,
            speaker=row.speaker,
            custom={"context": row.context} if row.context else None,
        )

        cut = MonoCut(
            id=seg_id,
            start=audio.offset,
            duration=audio.duration,
            channel=0,
            recording=rec,
            supervisions=[seg],
        )
        cuts.append(cut)

    print(f"Writing {len(cuts)} cuts to {output_cuts_path}...")
    CutSet.from_cuts(cuts).to_file(output_cuts_path)

    print(f"Successfully wrote Lhotse CutSet to: {output_cuts_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert jsonl manifest to lhotse cuts.")
    parser.add_argument("input_path", type=str, help="Input manifest")
    parser.add_argument("--output_path", type=str, default=None, help="Folder or file (.jsonl.gz)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()
    convert_manifest(args.input_path, args.output_path, force=args.force)
