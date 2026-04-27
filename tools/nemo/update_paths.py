import argparse
import logging

from ssak.utils.nemo_dataset import NemoDataset, resolve_manifest_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_paths(manifest_files, old_prefix, new_prefix):
    """Replace old_prefix with new_prefix in audio paths across manifests."""
    for mf in manifest_files:
        dataset = NemoDataset()
        data_type = dataset.load(str(mf))
        count = dataset.update_audio_paths(old_prefix, new_prefix)
        if count > 0:
            dataset.save(str(mf), data_type=data_type)
            logger.info(f"Updated {count} audio path(s) in {mf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update audio paths in NeMo manifests by replacing a prefix.",
        epilog="""Example:
  %(prog)s manifests/ --old-prefix /old/path --new-prefix /new/path
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Manifest files or directories")
    parser.add_argument("--old-prefix", required=True, help="String to find in audio paths")
    parser.add_argument("--new-prefix", required=True, help="String to replace with")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for manifest files (default: *.jsonl)")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")

    args = parser.parse_args()

    manifest_files = resolve_manifest_paths(args.inputs, pattern=args.pattern, recursive=args.recursive)
    if manifest_files:
        update_paths(manifest_files, args.old_prefix, args.new_prefix)

    logger.info("Done")
