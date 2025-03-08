#!/usr/bin/env python3

import os
import shutil
import subprocess

from tqdm import tqdm


def convert_video_to_audio(file_mp4, file_mp3):
    assert os.path.isfile(file_mp4)
    os.makedirs(os.path.dirname(file_mp3), exist_ok=True)
    subprocess.call(["ffmpeg", "-y", "-i", file_mp4, "-ar", "16000", "-ac", "1", file_mp3])
    assert os.path.isfile(file_mp3)


def extract_mp4(vid, file_mp4):
    vid = os.path.basename(os.path.splitext(vid)[0])
    CMD = [
        "youtube-dl",
        "--extract-audio",
        "--recode-video",
        "mp4",
        # "--audio-format", "mp3",
        "--id",
        f"https://www.youtube.com/watch?v={vid}",
    ]
    try:
        p = subprocess.Popen(CMD, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
    except Exception:
        raise RuntimeError(
            f"Failed to extract video using {CMD}. You may need to (re)install youtube-dl using:\n\
        sudo curl -L https://github.com/ytdl-patched/youtube-dl/releases/latest/download/youtube-dl -o /usr/local/bin/youtube-dl\n\
        sudo chmod a+rx /usr/local/bin/youtube-dl\n\
        "
        )
    output = f"{vid}.mp4"
    if not os.path.isfile(output):
        print(stdout.decode())
        print(stderr.decode())
        print(f"WARNING: Failed to extract {vid} using: {' '.join(CMD)}")
        return
    assert os.path.isfile(output), f"Failed to extract {vid} using {' '.join(CMD)}"
    shutil.move(output, file_mp4)
    assert os.path.isfile(file_mp4), f"Failed to extract {vid} to {file_mp4}"


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download mp4 video of a folder extracted with scrape_youtube.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Output folder path where audio and annotations will be saved (default: YouTubeFr, or YouTubeLang for another language than French).",
        type=str,
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.",
        type=str,
    )
    parser.add_argument("--video_ids", help="An explicit list of video ids.", type=str, default=None)
    # parser.add_argument('-v', '--verbose', help= "Print more information", action='store_true')
    args = parser.parse_args()

    path = os.path.realpath("YouTubeFr") if len(sys.argv) == 1 else sys.argv[1]

    lang = args.language
    path = args.path
    if not path:
        # YouTubeEn, YouTubeFr, etc.
        path = f"YouTube{lang[0].upper()}{lang[1:].lower()}"

    input_txt = f"{path}/{lang}"
    output_mp4 = f"{path}/mp4"
    output_mp3 = f"{path}/mp3"

    os.makedirs(output_mp4, exist_ok=True)

    if args.video_ids:
        if os.path.isfile(args.video_ids):
            with open(args.video_ids) as f:
                video_ids = [os.path.splitext(os.path.basename(line.strip()))[0] for line in f]
        elif os.path.isdir(args.video_ids):
            video_ids = [os.path.splitext(f)[0] for f in os.listdir(args.video_ids)]
        else:
            video_ids = args.video_ids.split(",")
    else:
        video_ids = os.listdir(input_txt)

    for id_ in tqdm(video_ids):
        id_ = os.path.splitext(id_)[0]
        file_mp4 = f"{output_mp4}/{id_}.mp4"
        file_mp3 = f"{output_mp3}/{id_}.mp3"
        if os.path.isfile(file_mp3):
            continue
        if not os.path.isfile(file_mp4) and not os.path.isfile(file_mp3):
            extract_mp4(id_, file_mp4)
            if not os.path.isfile(file_mp4):
                continue
        if not os.path.isfile(file_mp3):
            convert_video_to_audio(file_mp4, file_mp3)
