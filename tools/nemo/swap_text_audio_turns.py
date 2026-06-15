import argparse
import logging
from pathlib import Path

from ssak.utils.nemo_dataset import NemoDataset, resolve_manifest_paths, resolve_output_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def swap_first_two_turns(input_file, output_file, dataset_name=None):
    """Swap the first two turns of each row when the first is text and the second is audio.

    Warns (and leaves the row unchanged) for any other turn ordering.
    """
    dataset = NemoDataset(name=dataset_name)
    data_type = dataset.load(input_file, dataset_name=dataset_name)

    swapped = 0
    skipped = 0
    for row in dataset:
        turns = row.turns or []
        if len(turns) < 2:
            logger.warning(f"Row {row.id}: fewer than 2 turns, skipping")
            skipped += 1
            continue
        if turns[0].turn_type == "text" and turns[1].turn_type == "audio":
            turns[0], turns[1] = turns[1], turns[0]
            swapped += 1
        else:
            logger.warning(
                f"Row {row.id}: expected (text, audio) as first two turns, "
                f"got ({turns[0].turn_type}, {turns[1].turn_type}), skipping"
            )
            skipped += 1

    logger.info(f"{input_file}: swapped {swapped} rows, skipped {skipped} rows")
    dataset.save(output_file, data_type="multiturn" if data_type == "multiturn" else "asr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swap the first two turns in a NeMo manifest when they are (text, audio), making them (audio, text)."
    )
    parser.add_argument("input", help="Input manifest file (.jsonl), folder, or YAML config.", type=str)

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output_folder", help="Output folder.", type=str, default=None)
    output_group.add_argument("--output_file", help="Output file (only when input is a single .jsonl).", type=str, default=None)

    args = parser.parse_args()

    input_path = Path(args.input)
    input_lower = args.input.lower()
    is_single_file = (
        input_path.is_file()
        and input_path.suffix == ".jsonl"
        and not input_lower.endswith((".yaml", ".yml"))
        and "__OP_" not in args.input
    )

    if is_single_file:
        out_file = args.output_file or str(input_path.with_name(f"{input_path.stem}_audio_text.jsonl"))
        swap_first_two_turns(
            input_file=args.input,
            output_file=out_file,
            dataset_name=input_path.stem.replace("manifest_", ""),
        )
    else:
        if args.output_file:
            parser.error("When input is a folder, YAML, or shard pattern, use --output_folder.")
        manifests = resolve_manifest_paths(args.input)
        if not manifests:
            logger.warning(f"No manifest files found for input: {args.input}")
            exit(0)
        for mf in manifests:
            if args.output_folder:
                out_file = str(resolve_output_path(mf, args.input, args.output_folder))
            else:
                out_file = str(mf.parent.with_name(f"{mf.parent.name}_audio_text") / mf.name)
            dataset_name = mf.stem.replace("manifest_", "")
            try:
                swap_first_two_turns(input_file=str(mf), output_file=out_file, dataset_name=dataset_name)
            except Exception as e:
                raise Exception(f"Error in dataset '{dataset_name}' ({mf}): {e}") from e
