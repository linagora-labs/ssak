import argparse
import logging
import subprocess
from pathlib import Path

from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = [
 "nemo/ast/fr-en/context/CommonVoiceFR2EN/train.jsonl",
 "nemo/ast/en-fr/context/CommonVoiceEN2FR/train.jsonl",
 "nemo/ast/fr-es/context/CommonVoiceFR2ES/train.jsonl",
 "nemo/ast/fr-pt/context/CommonVoiceFR2PT/train.jsonl",
 "nemo/ast/de-fr/context/CommonVoiceDE2FR/train.jsonl",
 "nemo/ast/fr-it/context/CommonVoiceFR2IT/train.jsonl",
 "nemo/ast/fr-nl/context/CommonVoiceFR2NL/train.jsonl",
 "nemo/ast/fr-de/context/CommonVoiceFR2DE/train.jsonl",
 "nemo/ast/es-fr/context/CommonVoiceES2FR/train.jsonl",
 "nemo/ast/it-fr/context/CommonVoiceIT2FR/train.jsonl",
 "nemo/ast/pt-fr/context/CommonVoicePT2FR/train.jsonl",
 "nemo/ast/nl-fr/context/CommonVoiceNL2FR/train.jsonl"
]


def rsync_jsonl(datasets, source, destination, dry_run=False):
    """Rsync jsonl files from source to destination, preserving directory structure."""
    for ds in datasets:
        src_path = f"{source.rstrip('/')}/{ds}"
        dst_dir = Path(destination) / Path(ds).parent
        
        dst_dir.exists()
        
        cmd = ["rsync", "-rlDvz", "--size-only", "--mkpath"]
        if dry_run:
            cmd.append("--dry-run")
        cmd += [src_path, str(dst_dir)+"/"]

        logger.info(f"Rsyncing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"rsync failed for {ds} (exit code {result.returncode})")


def update_audio_paths(datasets, destination, old_prefix, new_prefix):
    """Replace old_prefix with new_prefix in all audio paths of the synced jsonl files."""
    old_prefix = old_prefix.rstrip("/") + "/"
    new_prefix = new_prefix.rstrip("/") + "/"

    for ds in datasets:
        jsonl_path = Path(destination) / ds
        if not jsonl_path.exists():
            logger.warning(f"Manifest not found, skipping: {jsonl_path}")
            continue

        dataset = NemoDataset()
        data_type = dataset.load(str(jsonl_path), show_progress_bar=ds)

        updated = 0
        for row in dataset.dataset:
            for turn in row.turns:
                if turn.turn_type == "audio" and turn.value.startswith(old_prefix):
                    turn.value = new_prefix + turn.value[len(old_prefix):]
                    updated += 1

        dataset.save(str(jsonl_path), data_type=data_type)
        logger.info(f"Updated {updated} audio paths in {ds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rsync NeMo dataset manifests and fix audio paths")
    parser.add_argument("--source", required=True, help="Rsync source (e.g. user@machineA:/data)")
    parser.add_argument("--destination", required=True, help="Local destination directory (e.g. /data)")
    parser.add_argument("--old-prefix", required=True, help="Audio path prefix to replace (e.g. /data/machineA)")
    parser.add_argument("--new-prefix", required=True, help="New audio path prefix (e.g. /data/machineB)")
    parser.add_argument("--datasets", nargs="*", help="Override the built-in DATASETS list")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to rsync (preview only)")
    parser.add_argument("--skip-rsync", action="store_true", help="Skip rsync, only update paths")

    args = parser.parse_args()
    datasets = args.datasets if args.datasets else DATASETS

    if not args.skip_rsync:
        rsync_jsonl(datasets, args.source, args.destination, dry_run=args.dry_run)

    if not args.dry_run:
        update_audio_paths(datasets, args.destination, args.old_prefix, args.new_prefix)

    logger.info("Done")
