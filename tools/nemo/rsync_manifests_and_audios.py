import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, resolve_manifest_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rsync_audios(manifest_files, destination, source="/", relative_to=None, dry_run=False, max_samples=None):
    """Rsync audio files referenced in manifests to destination."""
    audio_paths = set()
    processed_manifests = []

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
        logger.warning("No audio files found in manifests, nothing to sync")
        return processed_manifests

    if relative_to:
        prefix = relative_to.rstrip("/") + "/"
        file_paths = set()
        skipped_prefix = 0
        for p in audio_paths:
            if p.startswith(prefix):
                file_paths.add(p[len(prefix):])
            else:
                skipped_prefix += 1
        if skipped_prefix:
            logger.warning(f"{skipped_prefix} audio path(s) don't start with --relative-to '{prefix}', skipped")
        src = source.rstrip("/") + "/" + relative_to.strip("/") + "/"
    else:
        file_paths = set(audio_paths)
        src = source.rstrip("/") + "/"

    # Skip files that already exist at destination
    dest_dir = Path(destination)
    filtered_paths = []
    skipped = 0
    for fp in tqdm(file_paths, desc="Checking for existing files"):
        if (dest_dir / fp).exists():
            skipped += 1
        else:
            filtered_paths.append(fp)
    if skipped:
        logger.info(f"Skipped {skipped} files already present at destination")

    if not filtered_paths:
        logger.info("All files already synced, nothing to do")
        return processed_manifests

    logger.info(f"Syncing {len(filtered_paths)} files")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(sorted(filtered_paths)) + "\n")
        tmp_path = tmp.name

    try:
        cmd = f"rsync -rlDvz --size-only --copy-links --files-from={tmp_path} {src} {destination}"
        if dry_run:
            cmd += " --dry-run"
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            logger.error(f"rsync failed (exit code {result.returncode})")
    finally:
        Path(tmp_path).unlink()

    return processed_manifests


def _join_remote(prefix: str, rel: str) -> str:
    """Join a (possibly remote `host:/path`) prefix with a relative path without producing `//`."""
    rel = rel.lstrip("/")
    if not prefix or prefix == "/":
        return "/" + rel
    return prefix.rstrip("/") + "/" + rel


def rsync_manifests(manifest_paths, source, destination, relative_to=None, dry_run=False):
    """Rsync manifest JSONL files from source to destination.

    If `relative_to` is given, it is stripped from absolute inputs to compute
    the relative dataset path that is appended under `destination`. Otherwise
    inputs are treated as relative paths under `source` (legacy behavior).
    """
    for ds in manifest_paths:
        ds = str(ds)
        if relative_to:
            prefix = relative_to.rstrip("/") + "/"
            if ds.startswith(prefix):
                rel = ds[len(prefix):]
                src_path = ds
            elif not ds.startswith("/"):
                rel = ds
                src_path = _join_remote(source, rel)
            else:
                logger.warning(f"{ds} does not start with --relative-to '{prefix}', skipping")
                continue
        else:
            rel = ds.lstrip("/")
            src_path = _join_remote(source, rel)

        rel_parent = str(Path(rel).parent)
        if rel_parent in (".", ""):
            dst_dir = destination.rstrip("/") + "/"
        else:
            dst_dir = destination.rstrip("/") + "/" + rel_parent + "/"

        cmd = ["rsync", "-rlDvz", "--size-only", "--mkpath"]
        if dry_run:
            cmd.append("--dry-run")
        cmd += [src_path, dst_dir]

        logger.info(f"Rsyncing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"rsync failed for {ds} (exit code {result.returncode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rsync audio files and/or manifest JSONL files referenced in NeMo manifests.",
        epilog="""Examples:
  # Rsync audios referenced in manifests:
  %(prog)s manifests/ --rsync-audios /dest --relative-to /data/raw

  # Rsync manifest files from a remote machine:
  %(prog)s ds1.jsonl ds2.jsonl --rsync-manifests /local/data --rsync-source user@host:/data

  # Rsync manifests AND their referenced audios from a remote machine:
  %(prog)s nemo/asr/train.jsonl nemo/asr/test.jsonl --rsync-manifests /local/data --rsync-audios /local/data --rsync-source user@host:/data --relative-to /data/audio

  # Push a local dataset folder to a remote machine (local -> remote):
  %(prog)s /data/audio/nemo/asr/MyDataset --rsync-manifests host:/lustre/audio/ --rsync-audios host:/lustre/audio/ --relative-to /data/audio/ --dry-run

  # To update audio paths inside manifests after rsyncing, use tools/nemo/update_paths.py
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Manifest files, directories, or dataset relative paths (for --rsync-manifests)")

    action = parser.add_argument_group("Actions (at least one required)")
    action.add_argument("--rsync-audios", metavar="DEST", help="Rsync audio files referenced in manifests to this destination")
    action.add_argument("--rsync-manifests", metavar="DEST", help="Rsync manifest JSONL files to this destination")

    rsync_opts = parser.add_argument_group("Rsync options")
    rsync_opts.add_argument("--rsync-source", default="/", help="Rsync source prefix (default: /)")
    rsync_opts.add_argument("--relative-to", help="Strip this prefix from audio paths to preserve directory structure")
    rsync_opts.add_argument("--dry-run", action="store_true", help="Pass --dry-run to rsync (preview only)")
    rsync_opts.add_argument("--max-samples", type=int, help="Max samples per manifest; creates downsampled version if exceeded")

    discovery = parser.add_argument_group("Manifest discovery")
    discovery.add_argument("--pattern", default="*.jsonl", help="Glob pattern for manifest files (default: *.jsonl)")
    discovery.add_argument("--recursive", action="store_true", help="Search directories recursively")

    args = parser.parse_args()

    if not args.rsync_audios and not args.rsync_manifests:
        parser.error("Specify at least one action: --rsync-audios or --rsync-manifests")

    local_manifests = None
    if args.rsync_manifests:
        rsync_manifests(
            args.inputs, args.rsync_source, args.rsync_manifests,
            relative_to=args.relative_to, dry_run=args.dry_run,
        )
        # Find the rsynced manifests locally (only meaningful for local destinations)
        if ":" not in args.rsync_manifests:
            local_paths = [Path(args.rsync_manifests) / ds for ds in args.inputs]
            local_manifests = resolve_manifest_paths(local_paths, pattern=args.pattern, recursive=True)

    if args.rsync_audios:
        if local_manifests is not None:
            manifest_files = local_manifests
        else:
            manifest_files = resolve_manifest_paths(args.inputs, pattern=args.pattern, recursive=args.recursive)
        if manifest_files:
            rsync_audios(
                manifest_files, args.rsync_audios,
                source=args.rsync_source, relative_to=args.relative_to,
                dry_run=args.dry_run, max_samples=args.max_samples,
            )

    logger.info("Done")
