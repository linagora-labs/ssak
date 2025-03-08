import argparse
import os
import shutil

from tqdm import tqdm

from ssak.utils.kaldi import check_kaldi_dir


def load_segment_file(filename):
    data = []
    with open(filename) as file:
        for line in file:
            line = line.strip().split()
            seg = line[0]
            file = line[1]
            start = float(line[-2])
            end = float(line[-1])
            data.append({"seg": seg, "spk": seg.split("_")[0], "file": file, "start": start, "end": end})
        return data


def find_overlaps(data):
    data.sort(key=lambda x: (x["file"], x["start"]))
    overlaped_segments = set()
    for i, d in tqdm(enumerate(data), total=len(data)):
        if i > 0 and check_if_overlap({data[i - 1]["spk"]: data[i - 1]}, d, data[i + 1] if i + 1 < len(data) else None):
            overlaped_segments.add(d["seg"])
    return overlaped_segments


def check_if_can_concatenate(segment1, segment2, max_silence_duration_to_glue, max_segment_duration, segments_to_concatenate=None):
    if not segment1:
        return False
    if segment2["spk"] not in segment1:
        return False
    segment1 = segment1[segment2["spk"]]
    if segments_to_concatenate is not None and segment1["seg"] not in segments_to_concatenate and segment2["seg"] not in segments_to_concatenate:
        return False

    if segment1["file"] != segment2["file"]:
        return False
    elif segment1["spk"] != segment2["spk"]:
        return False
    elif abs(segment1["end"] - segment2["start"]) > max_silence_duration_to_glue:
        return False
    elif segment2["end"] - segment1["start"] > max_segment_duration:
        return False
    return True


def check_if_overlap(segment1, segment2, segment_next=None):
    if not segment1:
        return False
    segment1 = segment1.copy()
    # segment1.pop(segment2[segment2['spk']], None)
    if not segment1:
        return False
    for k in segment1:
        if segment1[k]["file"] == segment2["file"] and segment1[k]["end"] > segment2["start"]:
            return segment1[k]["seg"]

    if segment_next is None:
        return False
    if segment2["spk"] == segment_next["spk"] or segment2["file"] != segment_next["file"]:
        return False
    if segment2["end"] > segment_next["start"]:
        return segment_next["seg"]
    return False


def find_concatenate_segments(data, text, max_segment_duration=15, max_silence_duration_to_glue=0.5, segments_to_concatenate=None):
    data.sort(key=lambda x: (x["file"], x["start"]))
    previous_segment = dict()
    previous_text = dict()
    new_data = []
    new_text = []
    for i, d in tqdm(enumerate(data), total=len(data)):
        if check_if_can_concatenate(previous_segment, d, max_silence_duration_to_glue, max_segment_duration, segments_to_concatenate):
            spk = d["spk"]
            seg = previous_segment[spk]
            previous_segment[spk] = {
                "seg": seg["seg"] if d["seg"] not in segments_to_concatenate else d["seg"],
                "spk": seg["spk"],
                "file": seg["file"],
                "start": seg["start"],
                "end": d["end"],
            }
            previous_text[spk] += " " + text[d["seg"]]
        elif check_if_overlap(previous_segment, d, data[i + 1] if i + 1 < len(data) else None):
            if d["spk"] in previous_segment:
                new_data.append(previous_segment[d["spk"]])
                new_text.append({"seg": previous_segment[d["spk"]]["seg"], "text": previous_text[d["spk"]]})
                previous_segment.pop(d["spk"])
                previous_text.pop(d["spk"])
            previous_segment[d["spk"]] = d
            previous_text[d["spk"]] = text[d["seg"]]

        else:
            if previous_segment:
                for k in previous_segment:
                    new_data.append(previous_segment[k])
                    new_text.append({"seg": previous_segment[k]["seg"], "text": previous_text[k]})
                previous_segment = dict()
                previous_text = dict()
            previous_segment[d["spk"]] = d
            previous_text[d["spk"]] = text[d["seg"]]
    for k in previous_segment:
        new_data.append(previous_segment[k])
        new_text.append({"seg": previous_segment[k]["seg"], "text": previous_text[k]})
    return new_data, new_text


def load_file(filename):
    data = dict()
    with open(filename) as file:
        for line in file:
            line = line.strip().split()
            data[line[0]] = " ".join(line[1:])
        return data


def write_files(fsegment, ftext, new_segments_data, new_text_data):
    for seg, txt in zip(new_segments_data, new_text_data):
        fsegment.write(f"{seg['seg']} {seg['file']} {seg['start']} {seg['end']}\n")
        ftext.write(f"{txt['seg']} {txt['text']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glue overlapping segments to adjacent segments")
    parser.add_argument("input_folder", type=str, help="Kaldi folder containing segments and text files")
    parser.add_argument("--output_folder", type=str, default=None, help="Output folder (default: input_folder)")
    parser.add_argument(
        "--max_silence_duration_to_glue",
        type=float,
        default=0.5,
        help="Maximum silence duration between 2 segments to glue them",
    )
    parser.add_argument("--max_segment_duration", type=float, default=15, help="Maximum segment duration")
    parser.add_argument("--glue_mode", type=str, choices=["overlap", "all"], default="overlap", help="What to glue mode")
    args = parser.parse_args()

    input_folder = args.input_folder
    if args.output_folder is None:
        output_folder = input_folder
    else:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)

    data = load_segment_file(os.path.join(input_folder, "segments"))
    print(f"Number of segments: {len(data)}")
    text = load_file(os.path.join(input_folder, "text"))

    if args.glue_mode == "overlap":
        overlaped_segments = find_overlaps(data)
        print(f"Number of overlapping segments: {len(overlaped_segments)}")

    new_segments_data, new_text_data = find_concatenate_segments(
        data,
        text,
        max_segment_duration=args.max_segment_duration,
        max_silence_duration_to_glue=args.max_silence_duration_to_glue,
        segments_to_concatenate=None if args.glue_mode == "all" else overlaped_segments,
    )

    print(f"New number of segments: {len(new_segments_data)}")

    segment_filename = "segments" if input_folder != output_folder else "segments_new"
    text_filename = "text" if input_folder != output_folder else "text_new"
    with open(os.path.join(output_folder, segment_filename), "w") as fseg, open(os.path.join(output_folder, text_filename), "w") as ftext:
        write_files(fseg, ftext, new_segments_data, new_text_data)

    if output_folder != input_folder:
        shutil.copy(os.path.join(input_folder, "utt2spk"), os.path.join(output_folder, "utt2spk"))
        shutil.copy(os.path.join(input_folder, "wav.scp"), os.path.join(output_folder, "wav.scp"))
        if os.path.exists(os.path.join(input_folder, "spk2gender")):
            shutil.copy(os.path.join(input_folder, "spk2gender"), os.path.join(output_folder, "spk2gender"))
        check_kaldi_dir(output_folder)

    print(f"Done! Output folder: {output_folder}")
