#!/usr/bin/env python3

import os
import subprocess
import sys

from tqdm import tqdm


def convert_video_to_audio(video_path, audio_path):
    subprocess.call(["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file or folder")
    parser.add_argument("output", type=str, help="Output file or folder")
    args = parser.parse_args()

    args.input = sys.argv[1]
    args.output = sys.argv[2]

    if os.path.isfile(args.input):
        convert_video_to_audio(args.input, args.output)
    else:
        assert os.path.isdir(args.input), f"Failed to find {args.input} (not a file nor folder)"
        for filename in tqdm(os.listdir(args.input)):
            if filename.endswith(".mp4"):
                video = os.path.join(args.input, filename)
                outputfilename = os.path.splitext(filename)[0] + ".mp3"
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                audio = os.path.join(args.output, outputfilename)
                if os.path.isfile(audio):
                    continue
                convert_video_to_audio(video, audio)
