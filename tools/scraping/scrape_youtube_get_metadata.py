#!/usr/bin/env python3

import datetime
import os
import time
import urllib.error

import pytube

assert pytube.__version__ >= "15.0.0"

# def object_to_dict(obj):
#     if isinstance(obj, int|float|str):
#         return obj
#     elif isinstance(obj, list):
#         return [object_to_dict(x) for x in obj]
#     elif isinstance(obj, dict):
#         return {k: object_to_dict(v) for k, v in obj.items()}
#     # elif hasattr(obj,"__dict__"):
#     #     return object_to_dict(obj.__dict__)
#     else:
#         print(type(obj), [key for key in dir(obj) if not key.startswith('_')])
#         return object_to_dict(dict(
#             (key, eval(f"obj.{key}", globals(), locals())) for key in dir(obj) if not key.startswith('_')
#         ))


def yt_to_dict(
    x,
    level=10,
    simple_classes=[int, float, str, bool, type(None)],
    additional_attr=[
        "publish_date",
        "title",
        "author",
        "description",
        "metadata",
        "raw_metadata",
        "age_restricted",
        "vid_info",
    ],
    ignore_attr=[
        "_watch_html",
        "responseContext",
        "streamingData",
        "playbackTracking",
        "playerConfig",
        "trackingParams",
        "thumbnail",
        "backgroundability",
        "audioOnlyPlayability",
        "miniplayer",
        # "captions",
        "stream_monostate",
        "embed_url",
        "stream_monostate",
        "storyboards",
        "attestation",
    ],
    ignore_private=True,
):
    if max([isinstance(x, c) for c in simple_classes]):
        return x
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return dict((k, yt_to_dict(v, level - 1)) for k, v in x.items() if k not in ignore_attr and (not k.startswith("_") or not ignore_private))
    if isinstance(x, (list, tuple)):
        return [yt_to_dict(v, level - 1) for v in x]
    level -= 1
    if not hasattr(x, "__dict__"):
        params = {}
    elif level <= 0:
        params = dict((k, v) for k, v in x.__dict__.items() if max([isinstance(v, c) for c in simple_classes]) and k not in ignore_attr and (not k.startswith("_") or not ignore_private))
    else:
        params = dict((k, yt_to_dict(v, level - 1)) for k, v in x.__dict__.items() if k not in ignore_attr and (not k.startswith("_") or not ignore_private))
    for attr in additional_attr:
        if attr in dir(x):
            try:
                params[attr] = yt_to_dict(x.__getattribute__(attr))
            except pytube.exceptions.PytubeError as err:
                print(err)
                params[attr] = None
    # classname = str(type(x)).split("'")[1]
    return params  # | {"_class": classname}


def sort_dict(d):
    if isinstance(d, dict):
        return dict(sorted((k, sort_dict(v)) for k, v in d.items()))
    if isinstance(d, list):
        return [sort_dict(v) for v in d]
    return d


def not_empty(d):
    if isinstance(d, (str, list)):
        return bool(d)
    if isinstance(d, dict):
        return max([not_empty(v) for k, v in d.items() if not k.startswith("_")])
    return True


IDS_WITH_PROBLEMS = {}


def get_metadata(video_id, log_file=None):
    try:
        if isinstance(video_id, pytube.YouTube):
            yt = video_id
        else:
            video_id = os.path.splitext(os.path.basename(video_id))[0]
            yt = pytube.YouTube("https://www.youtube.com/watch?v=" + video_id)
            assert yt.video_id == video_id
        metadata = {
            "_video_id": yt.video_id,
            "_date": datetime.datetime.now().isoformat(),
        }
        try:
            yt.check_availability()
            metadata["_availability"] = "OK"
        except pytube.exceptions.VideoUnavailable:
            metadata["_availability"] = "ERROR"
        metadata.update(yt_to_dict(yt))
        status = metadata["vid_info"]["playabilityStatus"].get("status")
        error = None
        if status in ["UNPLAYABLE"]:
            url = metadata["watch_url"]
            if metadata["_availability"] == "OK":
                status = "OK"
            else:
                error = "Content restricted to members"
            if log_file:
                reported = error if error else "OK"
                add_in_file = url not in IDS_WITH_PROBLEMS
                if not add_in_file and IDS_WITH_PROBLEMS[url] != reported:
                    print("WARNING", url, "Problem changed", IDS_WITH_PROBLEMS[url], "=>", reported)
                    add_in_file = True
                if add_in_file:
                    with open(log_file, "a") as f:
                        f.write(f"{url}\t{reported}\n")
        if status != "OK" and not error:
            error = metadata["vid_info"]["playabilityStatus"].get("reason")
        metadata.update(
            {
                "_status": status,
                "_error": error,
            }
        )
        metadata = sort_dict(metadata)
        return metadata

    except urllib.error.HTTPError:
        print("HTTPError, retrying in 60 seconds...")
        time.sleep(60)
        return get_metadata(video_id, log_file=log_file)


def has_problems(metadata):
    return metadata["_status"] != "OK" or metadata["_error"]


def merge(dict_global, d, maximum=10):
    for k, v in d.items():
        # if k == "_js": print("Get", k, dict_global.get(k))
        if isinstance(v, dict):
            assert isinstance(dict_global.get(k, {}), dict)
            dict_global[k] = merge(dict_global.get(k, {}), v)
            # if k == "_js": print("Set", k, dict_global.get(k))
        elif k not in dict_global:
            dict_global[k] = [v]
            # if k == "_js": print("Set", k, dict_global.get(k))
        elif len(dict_global[k]) < maximum and v not in dict_global[k]:
            dict_global[k].append(v)
            # if k == "_js": print("Set", k, dict_global.get(k))
    return dict_global


def finalize(dict_global, max_string_len=28):
    if isinstance(dict_global, dict):
        dict_global = dict(dict_global)
        for k, v in dict_global.items():
            if isinstance(v, dict):
                dict_global[k] = finalize(v)
            elif isinstance(v, list):
                dict_global[k] = [finalize(vi) for vi in v]
                try:
                    dict_global[k] = list(set(dict_global[k]))
                except:
                    pass
                if len(dict_global[k]) == 1:
                    dict_global[k] = dict_global[k][0]

    elif isinstance(dict_global, str):
        if len(dict_global) > max_string_len:
            dict_global = dict_global[:max_string_len] + "..."

    return dict_global


if __name__ == "__main__":
    import argparse
    import json

    import tqdm

    parser = argparse.ArgumentParser("Get metadata from youtube videos")
    parser.add_argument("ids", nargs="*", help="Video ID, or file name as a video ID, or directory with files named as video IDs")
    parser.add_argument("--output", "-o", default=None, help="Output folder name")
    parser.add_argument("--ignore_if_exists", "-i", action="store_true", help="Ignore if output file exists")
    parser.add_argument(
        "--examples",
        "-p",
        default=None,
        help="An output file where will be outputed the list of possible values for each key",
    )
    parser.add_argument("--check", "-c", default=None, help="An output file where will be written ids to check")
    parser.add_argument("--max", "-m", default=None, help="Maximum number of video ids to process")
    args = parser.parse_args()

    if not args.ids:
        # Some interesting cases:
        args.ids = [
            # Good:
            "--QWtZrko9w",  # OK
            "--GikApuNNU",  # OK with streaming subtitles
            "kBF_NkwT768",  # available but marked as not playable
            # Bad:
            "32nkdvLq3oQ",  # "This video has been removed for violating YouTube's Terms of Service"
            "1UTk5IwG2vo",  # reserved to member
            "jnaXuppkbpk",  # private
            "JVzHx2w2IEI",  # private
            "FV06iXJj8q4",  # removed "This video has been removed by the uploader"
            "2w4fUaqJqzw",  # removed (?)
            "aaaaaaaaaaa",  # never existed
        ]

    WAIT = 0.3

    all_possible = {}

    video_ids = []
    for video_id in args.ids:
        if os.path.isdir(video_id):
            video_ids += [os.path.join(video_id, x) for x in os.listdir(video_id)]
        else:
            video_ids.append(video_id)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    if args.check and os.path.isfile(args.check):
        with open(args.check) as f:
            for line in f:
                id, status = line.strip().split("\t")
                IDS_WITH_PROBLEMS[id] = status

    total = 0
    problems = 0
    for video_id in tqdm.tqdm(video_ids):
        if args.output:
            output_filename = os.path.join(args.output, os.path.splitext(os.path.basename(video_id))[0] + ".json")
            if args.ignore_if_exists and os.path.exists(output_filename):
                continue

        metadata = get_metadata(video_id, log_file=args.check)

        total += 1
        if has_problems(metadata):
            problems += 1
            print(f"WARNING problem {problems}/{total} ({problems*100/total:.0f}%):\n{metadata['watch_url']}\n({metadata['_status']}: {metadata['_error']})")

        if args.output:
            with open(output_filename, "w") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        else:
            print(json.dumps(metadata, indent=4, ensure_ascii=False))

        all_possible = merge(all_possible, metadata)

        if args.examples:
            with open(args.examples, "w") as f:
                json.dump(finalize(all_possible), f, indent=4, ensure_ascii=False)

        if args.max and total >= int(args.max):
            break

        time.sleep(WAIT)
