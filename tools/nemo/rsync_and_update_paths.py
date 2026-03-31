import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

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


def rsync_audios(manifest_files, destination, source="/", relative_to=None, dry_run=False, max_samples=None):
    """Rsync audio files referenced in manifests to destination."""
    audio_paths = set()
    processed_manifests = []

    base = manifest_files[0].parent if len(manifest_files) == 1 else Path(manifest_files[0]).resolve().parent
    for mf in manifest_files:
        mf = Path(mf).resolve()
        dataset = NemoDataset()
        dataset.load(mf, show_progress_bar=str(mf.name))
        if max_samples and len(dataset.dataset) > max_samples:
            dataset.dataset = dataset.dataset[:max_samples]
            downsampled_path = mf.with_stem(f"{mf.stem}_downsampled_{max_samples}")
            dataset.save(downsampled_path)
            logger.info(f"Downsampled {mf} to {max_samples} samples -> {downsampled_path}")
            processed_manifests.append(downsampled_path)
        else:
            processed_manifests.append(mf)
        audio_paths.update(dataset.get_audio_paths(unique=True))

    logger.info(f"Found {len(audio_paths)} unique audio files across {len(processed_manifests)} manifest(s)")

    if not audio_paths:
        logger.warning("No audio files found in manifest, nothing to sync")
        return processed_manifests

    if relative_to:
        prefix = relative_to.rstrip("/") + "/"
        file_paths = set()
        for p in audio_paths:
            if p.startswith(prefix):
                file_paths.add(p[len(prefix):])
        src = source.rstrip("/") + "/" + relative_to.strip("/") + "/"
    else:
        file_paths = set(audio_paths)
        src = source.rstrip("/") + "/"

    # Skip files that already exist at destination
    dest_dir = Path(destination)
    filtered_paths = []
    skipped = 0
    for fp in tqdm(file_paths, desc="Checking for existing files"):
        local_path = dest_dir / fp
        if local_path.exists():
            skipped += 1
        else:
            filtered_paths.append(fp)
    if skipped:
        logger.info(f"Skipped {skipped} files already present at destination")
    file_paths = filtered_paths

    if not file_paths:
        logger.info("All files already synced, nothing to do")
        return processed_manifests

    logger.info(f"Syncing {len(file_paths)} files")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(sorted(file_paths)) + "\n")
        tmp_path = tmp.name

    try:
        dry_run_flag = " --dry-run" if dry_run else ""
        cmd = f"rsync -rlDvz --size-only --copy-links{dry_run_flag} --files-from={tmp_path} {src} {destination}"
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            logger.error(f"rsync failed (exit code {result.returncode})")
    finally:
        Path(tmp_path).unlink()

    return processed_manifests


def rsync_manifests(manifest_paths, source, destination, dry_run=False):
    """Rsync manifest JSONL files from source to destination."""
    for ds in manifest_paths:
        ds = str(ds)
        src_path = f"{source.rstrip('/')}/{ds}"
        dst_dir = Path(destination) / Path(ds).parent

        cmd = ["rsync", "-rlDvz", "--size-only", "--mkpath"]
        if dry_run:
            cmd.append("--dry-run")
        cmd += [src_path, str(dst_dir) + "/"]

        logger.info(f"Rsyncing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"rsync failed for {ds} (exit code {result.returncode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rsync audio files / manifests and/or update audio paths in NeMo manifests",
        epilog="""Examples:
  # Pure path update (replaces update_paths.py):
  %(prog)s manifests/ --old-prefix /old/path --new-prefix /new/path

  # Rsync audios + update paths (replaces rsync_audios.py):
  %(prog)s manifests/ --rsync-audios /dest --relative-to /data/raw --update-paths --old-prefix /old --new-prefix /new

  # Rsync manifest files + update paths (replaces rsync_datasets.py):
  %(prog)s ds1.jsonl ds2.jsonl --rsync-manifests user@host:/data --rsync-source user@host:/data --old-prefix /old --new-prefix /new
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Manifest files, directories, or dataset relative paths (for --rsync-manifests)")

    # Actions
    action = parser.add_argument_group("Actions (at least one required)")
    action.add_argument("--rsync-audios", metavar="DEST", help="Rsync audio files referenced in manifests to this destination")
    action.add_argument("--rsync-manifests", metavar="DEST", help="Rsync manifest JSONL files to this destination")
    action.add_argument("--update-paths", action="store_true", help="Update audio paths in manifests (requires --old-prefix and --new-prefix)")

    # Path replacement
    paths = parser.add_argument_group("Path replacement")
    paths.add_argument("--old-prefix", help="String to find in audio paths")
    paths.add_argument("--new-prefix", help="String to replace with")

    # Rsync options
    rsync_opts = parser.add_argument_group("Rsync options")
    rsync_opts.add_argument("--rsync-source", default="/", help="Rsync source prefix (default: /)")
    rsync_opts.add_argument("--relative-to", help="Strip this prefix from audio paths to preserve directory structure")
    rsync_opts.add_argument("--dry-run", action="store_true", help="Pass --dry-run to rsync (preview only)")
    rsync_opts.add_argument("--max-samples", type=int, help="Max samples per manifest; creates downsampled version if exceeded")

    # Manifest discovery
    discovery = parser.add_argument_group("Manifest discovery")
    discovery.add_argument("--pattern", default="*.jsonl", help="Glob pattern for manifest files (default: *.jsonl)")
    discovery.add_argument("--recursive", action="store_true", help="Search directories recursively")

    args = parser.parse_args()

    # Validate: at least one action
    has_path_update = args.old_prefix and args.new_prefix
    if not args.rsync_audios and not args.rsync_manifests and not has_path_update:
        parser.error("Specify at least one action: --rsync-audios, --rsync-manifests, or --old-prefix/--new-prefix for path update")

    if args.update_paths and not (args.old_prefix and args.new_prefix):
        parser.error("--update-paths requires --old-prefix and --new-prefix")

    # Rsync manifests (inputs are relative dataset paths, not local files)
    if args.rsync_manifests:
        rsync_manifests(args.inputs, args.rsync_source, args.rsync_manifests, dry_run=args.dry_run)
        if has_path_update and not args.dry_run:
            # After rsync, manifests are at destination — update paths there
            dest_manifests = [Path(args.rsync_manifests) / ds for ds in args.inputs]
            update_paths([p for p in dest_manifests if p.exists()], args.old_prefix, args.new_prefix)

    # Rsync audios (inputs are local manifest files/dirs)
    if args.rsync_audios:
        manifest_files = resolve_manifest_paths(args.inputs, pattern=args.pattern, recursive=args.recursive)
        if manifest_files:
            processed = rsync_audios(
                manifest_files, args.rsync_audios,
                source=args.rsync_source, relative_to=args.relative_to,
                dry_run=args.dry_run, max_samples=args.max_samples,
            )
            if args.update_paths and not args.dry_run:
                update_paths(processed, args.old_prefix, args.new_prefix)

    # Pure path update (no rsync)
    if has_path_update and not args.rsync_audios and not args.rsync_manifests:
        manifest_files = resolve_manifest_paths(args.inputs, pattern=args.pattern, recursive=args.recursive)
        if manifest_files:
            update_paths(manifest_files, args.old_prefix, args.new_prefix)

    logger.info("Done")
