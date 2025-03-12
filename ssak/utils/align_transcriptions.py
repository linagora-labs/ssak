# Source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from ssak.infer.general import (
    compute_log_probas,
    decode_log_probas,
    get_model_sample_rate,
    get_model_vocab,
    load_model,
)
from ssak.utils.misc import hashmd5
from ssak.utils.text_basic import transliterate
from ssak.utils.text_basic import _punctuation
from ssak.utils.viewer import PlayWav

imshow_opts = dict(origin="upper", aspect="auto", vmax=0)  # vmin = -25,

USE_MAX = True
USE_CHAR_REPEATED = True


def get_trellis(emission, tokens, blank_id=0, first_as_garbage=False):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(emission.device)
    trellis[0, 0] = 0
    if first_as_garbage:
        trellis[1:, 0] = (1 - emission[:, tokens[0]].exp()).log()
    else:
        trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)

    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        if USE_CHAR_REPEATED:
            trellis[t + 1, 1:] = (
                torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    torch.maximum(
                        trellis[t, 1:] + emission[t, tokens],
                        # Score for changing to the next token
                        trellis[t, :-1] + emission[t, tokens],
                    ),
                )
                if USE_MAX
                else torch.logaddexp(
                    trellis[t, 1:] + emission[t, blank_id],
                    torch.logaddexp(trellis[t, 1:] + emission[t, tokens], trellis[t, :-1] + emission[t, tokens]),
                )
            )
        else:
            trellis[t + 1, 1:] = (
                torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
                if USE_MAX
                else torch.logaddexp(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
            )

    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        if USE_CHAR_REPEATED:
            if USE_MAX:
                stayed = torch.maximum(stayed, trellis[t - 1, j] + emission[t - 1, tokens[j - 1]])
            else:
                stayed = torch.logaddexp(stayed, trellis[t - 1, j] + emission[t - 1, tokens[j - 1]])
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        if USE_CHAR_REPEATED and changed < stayed and t < emission.shape[0]:
            if USE_MAX:
                prob = torch.maximum(emission[t - 1, 0], emission[t, tokens[j - 1]]).exp().item()
            else:
                prob = torch.logaddexp(emission[t - 1, 0], emission[t, tokens[j - 1]]).exp().item()
        else:
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise RuntimeError("Failed to align (not enough tokens for the duration?)")
    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(transcript, path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments, separator=" "):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


# Plotting functions


def plot_trellis_with_path(trellis, path, ax=plt):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index + 1, p.token_index + 1] = float("nan")
    ax.imshow(trellis_with_path[1:, :].T, **imshow_opts)


def plot_trellis_with_segments(trellis, segments, transcript, path, plot_spaces=False, blank_first=True):
    if blank_first:
        transcript = ["_"] + list(transcript)

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
    ax1.set_title("Path, label and probability for each label")
    plot_trellis_with_path(trellis, path, ax1)
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        if blank_first:
            i += 1
        ax1.annotate(seg.label, (seg.start, i), weight="bold", verticalalignment="center", horizontalalignment="right")

    ax2.set_title("Label probability with and without repetition")
    xs, hs, ws = [], [], []
    for seg in segments:
        if not plot_spaces and seg.label == " ":
            continue
        xs.append((seg.end + seg.start) / 2 - 0.5)
        hs.append(seg.score)
        ws.append(seg.end - seg.start)
        ax2.annotate(seg.label, (seg.start, -0.07), weight="bold")
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if not plot_spaces and label == " ":
            continue
        xs.append(p.time_index)
        hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-0.1, 1.1)


def plot_alignments(trellis, segments, word_segments, waveform, sample_rate=16000, wav_file=None, emission=None, labels=None):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != " ":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, axes = plt.subplots(3 if emission is not None else 2, figsize=(16, 9.5))
    [ax1, ax2] = axes[-2:]
    plt.tight_layout()

    if emission is not None:
        ax0 = axes[0]
        if labels is not None and len(labels) < emission.shape[-1]:
            emission = emission[:, : len(labels)]
        ax0.imshow(emission.T, **imshow_opts)
        # ax0.set_ylabel("Labels")
        # ax0.set_yticks([])
        if labels is not None:
            new_labels = []
            i = 0
            for l in labels:
                if l == " ":
                    l = "SPACE"
                elif l.startswith("<"):
                    l = l.upper()
                else:
                    l = l + " " * (i % 3) * 2
                    i += 1
                new_labels.append(l)
            ax0.set_yticks(range(len(labels)), labels=new_labels)

    ax1.imshow(trellis_with_path[1:, :].T, **imshow_opts)
    ax1.set_xticks([])
    transcript = [s.label for s in segments]
    if transcript is not None:
        ax1.set_yticks(range(len(transcript) + 1), labels=["_"] + list(transcript))
    else:
        ax1.set_yticks([])

    for word in word_segments:
        ax1.axvline(word.start - 0.5)
        ax1.axvline(word.end - 0.5)

    for i, seg in enumerate(segments):
        ax1.annotate(seg.label, (seg.start, i + 1), weight="bold", verticalalignment="center", horizontalalignment="right")
        ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 1), fontsize=8)

    # The original waveform
    ratio = len(waveform) / emission.size(0)
    ax2.plot(waveform)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

    for seg in segments:
        label = seg.label if seg.label not in ["|", " "] else "\u2423"  # space
        ax2.annotate(label, (seg.start * ratio, 0.9))
    xticks = ax2.get_xticks()
    ax2.set_xticks(xticks, xticks / sample_rate)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, len(waveform))

    if wav_file:
        PlayWav(wav_file, ax2, draw=False)


# Main function


def compute_alignment(
    audio,
    transcript,
    model,
    add_before_after=None,
    first_as_garbage=False,
    plot=False,
    verbose=False,
):
    emission = compute_log_probas(model, audio)

    if transcript is None:
        transcript = decode_log_probas(model, emission)
        print("Transcript:", transcript)

    if isinstance(transcript, str):
        transcript_characters = transcript
        transcript_words = None
    else:
        assert isinstance(transcript, list), f"Got unexpected transcript (of type {type(transcript)})"
        for i, w in enumerate(transcript):
            assert isinstance(w, str), f"Got unexpected type {type(w)} (not a string)"
            # if w.strip() != w:
            #     print(f"WARNING: Got a word starting or ending with a space: '{w}'")
            #     transcript[i] = w.strip()
        transcript_characters = " ".join(transcript)
        transcript_words = transcript

    if plot > 1:
        plt.imshow(emission.T, **imshow_opts)
        plt.colorbar()
        plt.title("Frame-wise class probability")
        plt.xlabel("Time")
        plt.ylabel("Labels")
        plt.show()

    labels, blank_id = get_model_vocab(model)
    space_id = blank_id
    if " " in labels:
        space_id = labels.index(" ")

    if add_before_after:
        assert len(add_before_after) == 1, f"The character to add before and after the transcript must be a single character ({add_before_after} is not a single character)"
        assert add_before_after in labels, f"The character to add before and after the transcript must be in the model vocabulary ({add_before_after} not in {labels})"
        transcript_characters = add_before_after + transcript_characters + add_before_after

    labels = labels[: emission.shape[1]]
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [loose_get_char_index(dictionary, c, space_id) for c in transcript_characters]
    tokens = [i for i in tokens if i is not None]

    trellis = get_trellis(emission, tokens, blank_id=blank_id, first_as_garbage=first_as_garbage)

    if plot > 1:
        plt.imshow(trellis.T, **imshow_opts)
        plt.colorbar()
        plt.show()

    path = backtrack(trellis, emission, tokens, blank_id=blank_id)

    if plot > 1:
        plot_trellis_with_path(trellis, path)
        plt.title("The path found by backtracking")
        plt.show()

    char_segments = merge_repeats(transcript_characters, path)

    if add_before_after:
        assert char_segments[0].label == add_before_after
        assert char_segments[-1].label == add_before_after
        char_segments = char_segments[1:-1]
        trellis = trellis[:, [0] + list(range(2, trellis.shape[1] - 1))]
        transcript_characters = transcript_characters[1:-1]

    if transcript_words is None:
        word_segments = merge_words(char_segments)
    else:
        word_segments = []
        i2 = -1
        for word in transcript_words:
            i1 = i2 + 1
            i2 = i1 + len(word)
            segs1 = char_segments[i1:i2]
            word_check = "".join([seg.label for seg in segs1])
            assert word_check == word
            segs2 = [s for s in segs1 if s.label not in " " + _punctuation]
            if len(segs2) != 0:
                segs = segs2
            else:
                segs = segs1
            score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
            word_segments.append(Segment(word, segs[0].start, segs[-1].end, score))
            if verbose:
                for s in segs1:
                    if s == segs[0]:
                        print(s.label, s.start, s.end, word, word_segments[-1].start, word_segments[-1].end)
                    else:
                        print(s.label, s.start, s.end)

    if plot:
        plot_trellis_with_segments(trellis, char_segments, transcript_characters, path)
        plt.axvline(word_segments[0].start - 0.5, color="black")
        plt.axvline(word_segments[-1].end - 0.5, color="black")
        plt.tight_layout()
        plt.show()

    return labels, emission, trellis, char_segments, word_segments


MISSING_LABELS = {}


def loose_get_char_index(dictionary, c, default):
    global MISSING_LABELS
    i = dictionary.get(c, None)
    if i is None:
        other_char = list(set([c.lower(), c.upper(), transliterate(c), transliterate(c).lower(), transliterate(c).upper()]))
        for c2 in other_char:
            i = dictionary.get(c2, None)
            if i is not None:
                break
        if i is None:
            key = hashmd5(dictionary)
            if key not in MISSING_LABELS:
                MISSING_LABELS[key] = []
            if c not in MISSING_LABELS[key]:
                print("WARNING: cannot find label " + " / ".join(list(set([c] + other_char))))
                MISSING_LABELS[key].append(c)
            i = default
    return i


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Show the alignment of a given audio file and transcription",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", help="Input model folder or name (Transformers, Speechbrain)", type=str)
    parser.add_argument("audio", help="Input audio files", type=str)
    # optional arguments
    parser.add_argument(
        "transcription",
        help="Audio Transcription. If not provided, the automatic transcription from the model will be used.",
        type=str,
        default=[],
        nargs="*",
    )
    parser.add_argument("--plot_intermediate", help="To make intermediate plots.", default=False, action="store_true")
    args = parser.parse_args()

    from ssak.utils.audio import load_audio

    audio_path = args.audio
    transcript = " ".join(args.transcription)
    if not transcript:
        transcript = None

    model = load_model(args.model)
    sample_rate = get_model_sample_rate(model)

    audio = load_audio(audio_path, sample_rate=sample_rate)

    labels, emission, trellis, segments, word_segments = compute_alignment(audio, transcript, model, plot=args.plot_intermediate)

    del model

    plot_alignments(
        trellis,
        segments,
        word_segments,
        audio,
        sample_rate=sample_rate,
        wav_file=audio_path,
        emission=emission,
        labels=labels,
    )
    plt.show()
