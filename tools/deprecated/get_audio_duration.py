#!/usr/bin/env python3

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get duration of audio file(s)")
    parser.add_argument("input_file", type=str, help="Input audio file or wav.scp file")
    parser.add_argument("dest", type=str, help="Output file", nargs="?")
    args = parser.parse_args()

    input_file = args.input_file

    import os

    assert os.path.isfile(input_file), f"Input file not found: {input_file}"

    import sys

    from ssak.utils.audio import get_audio_duration
    from ssak.utils.kaldi import parse_kaldi_wavscp

    with open(args.dest, "w") if args.dest else sys.stdout as fout:
        if input_file.endswith(".scp"):
            waves = parse_kaldi_wavscp(input_file)
            for wavid, path in waves.items():
                assert os.path.isfile(path), f"File not found: {path}"
                duration = get_audio_duration(path)
                fout.write(f"{wavid} {duration:.3f}\n")
        else:
            duration = get_audio_duration(input_file)
            fout.write(f"{input_file} {duration:.3f}\n")
