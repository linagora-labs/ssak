import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rsync audio files referenced in a NeMo manifest")
    parser.add_argument("manifest", help="Path to a NeMo manifest JSONL file or a directory containing manifests")
    parser.add_argument("destination", help="Rsync destination (local path or user@host:/path)")
    parser.add_argument("--source", default="/", help="Rsync source (e.g. user@host:/). Defaults to /.")
    parser.add_argument("--relative-to", help="Strip this prefix from audio paths to preserve directory structure after it (e.g. /datasets/audio/raw/transcript)")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern to filter manifest files when using a directory (default: *.jsonl, e.g. test.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to rsync (preview only)")
    parser.add_argument("--update-paths", action="store_true", help="Update audio paths in manifests to point to destination after successful rsync")
    parser.add_argument("--max-samples", type=int, help="Max number of samples per manifest. Creates a downsampled manifest if exceeded.")

    args = parser.parse_args()

    # Collect manifest files
    manifest_path = Path(args.manifest)
    if manifest_path.is_dir():
        manifest_files = sorted(str(p) for p in manifest_path.rglob(args.pattern))
        logger.info(f"Found {len(manifest_files)} manifest files matching '{args.pattern}' in {args.manifest}")
        if not manifest_files:
            logger.warning(f"No files matching '{args.pattern}' found in directory")
            exit(0)
    else:
        manifest_files = [args.manifest]

    # Load all manifests and collect unique audio paths
    audio_paths = set()
    downsampled_manifest_files = []
    base = Path(args.manifest).resolve()
    for mf in manifest_files:
        mf = Path(mf).resolve()
        relative = mf.relative_to(base)
        dataset = NemoDataset()
        dataset.load(mf, dataset_name=relative)
        if args.max_samples and len(dataset.dataset) > args.max_samples:
            dataset.dataset = dataset.dataset[:args.max_samples]
            downsampled_path = str(mf.with_stem(f"{mf.stem}_downsampled_{args.max_samples}"))
            dataset.save(downsampled_path)
            logger.info(f"Downsampled {mf} to {args.max_samples} samples -> {downsampled_path}")
            downsampled_manifest_files.append(downsampled_path)
        else:
            downsampled_manifest_files.append(mf)
        audio_paths.update(dataset.get_audio_paths(unique=True))
    manifest_files = downsampled_manifest_files
    logger.info(f"Found {len(audio_paths)} unique audio files across {len(manifest_files)} manifest(s)")

    if not audio_paths:
        logger.warning("No audio files found in manifest, nothing to sync")
        exit(0)

    # Determine source and file list for rsync
    if args.relative_to:
        # Strip prefix from audio paths so destination preserves structure after it.
        # Append the prefix to --source so rsync still reads from the right place.
        prefix = args.relative_to.rstrip("/") + "/"
        file_paths = set()
        for p in audio_paths:
            if p.startswith(prefix):
                file_paths.add(p[len(prefix):])
            else:
                # logger.warning(f"Path does not start with '{prefix}', keeping as-is: {p}")
                # file_paths.add(p)
                pass
        src = args.source.rstrip("/") + "/" + args.relative_to.strip("/") + "/"
    else:
        raise NotImplementedError("Not implemented yet")
        # file_paths = set(audio_paths)
        # src = args.source.rstrip("/") + "/"

    # Skip files that already exist at destination
    dest_dir = Path(args.destination)
    filtered_paths = []
    skipped = 0
    for fp in tqdm(file_paths, desc="Checking for existing files"):
        local_path = dest_dir / fp
        if local_path.exists():
            logger.debug(f"Skipping already synced: {local_path}")
            skipped += 1
        else:
            filtered_paths.append(fp)
    if skipped:
        logger.info(f"Skipped {skipped} files already present at destination")
    file_paths = filtered_paths

    if not file_paths:
        logger.info("All files already synced, nothing to do")
        exit(0)

    # Write file list and run rsync
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(sorted(file_paths)) + "\n")
        tmp_path = tmp.name

    try:
        dry_run_flag = " --dry-run" if args.dry_run else ""
        cmd = f"rsync -rlDvz --size-only --copy-links{dry_run_flag} --files-from={tmp_path} {src} {args.destination}"
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
    finally:
        Path(tmp_path).unlink()

    # Update manifest paths after successful rsync (not dry-run)
    if args.update_paths and not args.dry_run and result.returncode == 0:
        dest = Path(args.destination)
        for mf in manifest_files:
            dataset = NemoDataset()
            dataset.load(mf)
            for row in dataset.dataset:
                for turn in row.turns:
                    if turn.turn_type == "audio":
                        if args.relative_to:
                            prefix = args.relative_to.rstrip("/") + "/"
                            if turn.value.startswith(prefix):
                                turn.value = str(dest / turn.value[len(prefix):])
                        else:
                            turn.value = str(dest / turn.value.lstrip("/"))
            dataset.save(mf)
        logger.info(f"Updated audio paths in {len(manifest_files)} manifest(s)")

    logger.info("Done")
