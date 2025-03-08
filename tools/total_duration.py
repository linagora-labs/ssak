#!/usr/bin/env python3


from ssak.utils.audio import get_audio_total_duration


def second2time(val):
    if val == float("inf"):
        return "_"
    # Convert seconds to time
    hours = int(val // 3600)
    minutes = int((val % 3600) // 60)
    seconds = int(val % 60)
    milliseconds = int((val % 1) * 1000)
    s = f"{seconds:02d}.{milliseconds:03d}"
    if True:  # hours > 0 or minutes > 0:
        s = f"{minutes:02d}:{s}"
    if True:  # hours > 0:
        s = f"{hours:02d}:{s}"
    return s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get duration of audio file(s)")
    parser.add_argument("input", type=str, help="Input files or folders", nargs="+")
    args = parser.parse_args()

    nb, duration = get_audio_total_duration(args.input, verbose=True)

    print(f"Total Duration of {nb} files: {second2time(duration)}")
