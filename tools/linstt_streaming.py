#!/usr/bin/env python3

from __init__ import *

from ssak.utils.linstt import linstt_streaming

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe input streaming (from mic) with LinSTT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server",
        help="Transcription server",
        default="wss://api.linto.ai/stt-vivatech-streaming/streaming",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--audio_file", default=None, help="A path to an audio file to transcribe (if not provided, use mic)")
    args = parser.parse_args()

    res = linstt_streaming(args.audio_file, args.server, verbose=2 if args.verbose else 1)
