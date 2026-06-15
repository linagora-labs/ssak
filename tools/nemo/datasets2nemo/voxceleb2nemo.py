"""Convert VoxCeleb1 + VoxCeleb2 into NeMo speaker-verification manifests.

VoxCeleb has no transcripts (the per-clip `txt/` files are face-tracking boxes),
so the only exploitable ground truth is the *speaker identity*, which lives in the
path:  <root>/<split>/<aac|wav>/<speaker_id>/<youtube_video_id>/<utt>.wav
                                  ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^  ^^^^^
                                  speaker       session (video)    segment

For a voice<->text multimodal model we turn that into a *speaker verification*
task: each manifest row is a pair of clips (A, B) plus a text instruction, and the
target text is "yes"/"no" — same speaker or not. No audio is concatenated; each
clip is referenced as-is via its own audio turn.

Expected layout (standard VoxCeleb download already on disk):

    VoxCeleb1/dev/wav/<id>/<vid>/<utt>.wav      VoxCeleb1/test/wav/<id>/<vid>/<utt>.wav
    VoxCeleb2/dev/aac/<id>/<vid>/<utt>.wav      VoxCeleb2/test/aac/<id>/<vid>/<utt>.wav

(VC2 stores re-encoded .wav under the `aac/` dir; `combined.wav` files added by
other tooling are ignored.)

Splits produced:
  - train + dev : drawn from VC1 dev + VC2 dev, split *by speaker* so dev speakers
                  never appear in train (no speaker leakage).
  - test        : drawn from VC1 test + VC2 test.

Pairs are balanced 50/50. Positive pairs prefer two *different* videos of the same
speaker (cross-session — the non-trivial case); negative pairs are two distinct
speakers. Everything is deterministic given --seed.
"""

import argparse
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "VoxCeleb"

# Source roots. Each entry: (dataset tag, split dir, audio subdir).
SOURCES = {
    "train_dev": [
        ("VoxCeleb2", "dev", "aac"),
        ("VoxCeleb1", "dev", "wav"),
    ],
    "test": [
        ("VoxCeleb2", "test", "aac"),
        ("VoxCeleb1", "test", "wav"),
    ],
}

# Instruction prompts (the User text turn). One is drawn at random per row so the
# model does not overfit a single phrasing. The two audio turns follow the prompt.
# Prompts come in several languages; the yes/no answer is given in the same language
# as the prompt. (The audio itself is English — only the instruction varies.)
# Each prompt has a *polarity*: "same" prompts ask whether it is the same speaker
# (answer = yes when same); "diff" prompts ask whether the speakers are different
# (answer = yes when NOT the same). This keeps the model from binding "same speaker"
# to a fixed answer token.
PROMPT_BANK = {
    "en": {
        "yes": "yes",
        "no": "no",
        # Appended to the question ONLY in the yes/no style (one drawn at random); the
        # chat style keeps the bare question.
        "answer_suffix": [
            "Answer yes or no.",
            "Yes or no?",
            "Reply with yes or no.",
            "Answer with yes or no.",
            "Respond yes or no.",
            "Just say yes or no.",
            "Answer only yes or no.",
            "Please reply yes or no.",
            "A simple yes or no will do.",
        ],
        # Full-sentence answers (chat style): "<lead>, <clause>". The lead (Yes/No)
        # matches the question polarity; the clause states the ground-truth fact.
        "yes_lead": "Yes",
        "no_lead": "No",
        "same_clause": [
            "both clips are spoken by the same person.",
            "these two recordings are from the same speaker.",
            "it is the same speaker in both clips.",
            "the same voice is heard in both recordings.",
            "both segments feature one and the same individual.",
            "the two utterances were produced by the same person.",
        ],
        "diff_clause": [
            "these are two different speakers.",
            "the two clips are spoken by different people.",
            "the voices belong to different persons.",
            "each recording features a different speaker.",
            "the two utterances come from two distinct individuals.",
            "the speakers in the two clips are not the same.",
        ],
        "same": [
            "You are given two audio clips. Are they spoken by the same person?",
            "Listen to the two recordings. Is it the same speaker in both?",
            "Do these two audio segments come from the same speaker?",
            "Compare the two voices. Are they the same person?",
            "Two utterances are provided. Were they said by the same speaker?",
            "Decide if both clips feature the same speaker.",
            "Here are two recordings. Is the speaker the same?",
            "Are both audio samples from one and the same person?",
            "Determine whether the two voices belong to the same individual.",
            "Two voice samples. Is the same person talking in both?",
            "Tell me whether both clips were recorded by the same speaker.",
            "Having heard both, would you say it is one single speaker?",
            "Is the person speaking in the first clip the same as in the second?",
            "Check the two recordings: do they share the same speaker?",
            "Given these two utterances, is the talker identical in each?",
        ],
        "diff": [
            "You are given two audio clips. Are they spoken by different people?",
            "Listen to the two recordings. Are these two different speakers?",
            "Do these two audio segments come from different speakers?",
            "Compare the two voices. Are they two different persons?",
            "Decide whether the two clips feature different speakers.",
            "Are these two recordings from distinct people?",
            "Determine whether the two voices belong to different individuals.",
            "Two voice samples. Are different people talking?",
            "Tell me whether the two clips were recorded by different speakers.",
            "Having heard both, would you say these are two separate speakers?",
            "Is the person in the first clip different from the one in the second?",
            "Check the two recordings: do they have different speakers?",
            "Given these two utterances, are the talkers distinct?",
        ],
    },
    "fr": {
        "yes": "oui",
        "no": "non",
        "answer_suffix": [
            "Réponds oui ou non.",
            "Oui ou non ?",
            "Réponds simplement par oui ou non.",
            "Réponds par oui ou par non.",
            "Dis simplement oui ou non.",
            "Réponds uniquement par oui ou non.",
            "Un simple oui ou non suffit.",
            "Contente-toi de répondre oui ou non.",
            "Réponds seulement par oui ou par non.",
        ],
        "yes_lead": "Oui",
        "no_lead": "Non",
        "same_clause": [
            "les deux extraits sont prononcés par la même personne.",
            "c'est le même locuteur dans les deux enregistrements.",
            "les deux clips proviennent du même locuteur.",
            "on entend la même voix dans les deux enregistrements.",
            "les deux segments sont d'une seule et même personne.",
            "les deux énoncés ont été produits par la même personne.",
        ],
        "diff_clause": [
            "ce sont deux locuteurs différents.",
            "les deux extraits sont prononcés par des personnes différentes.",
            "les voix appartiennent à des personnes différentes.",
            "chaque enregistrement présente un locuteur différent.",
            "les deux énoncés proviennent de deux personnes distinctes.",
            "les locuteurs des deux extraits ne sont pas les mêmes.",
        ],
        "same": [
            "Voici deux segments audio. Sont-ils prononcés par la même personne ?",
            "Écoute les deux enregistrements. Est-ce le même locuteur dans les deux ?",
            "Ces deux segments audio proviennent-ils du même locuteur ?",
            "Compare les deux voix. S'agit-il de la même personne ?",
            "Deux énoncés sont fournis. Ont-ils été dits par le même locuteur ?",
            "Détermine si les deux extraits sont de la même personne.",
            "Le locuteur est-il le même dans ces deux extraits ?",
            "Les deux échantillons audio sont-ils d'une seule et même personne ?",
            "Voici deux échantillons de voix. Est-ce la même personne qui parle ?",
            "Dis-moi si les deux extraits ont été enregistrés par le même locuteur.",
            "Après les avoir écoutés, dirais-tu qu'il s'agit d'un seul locuteur ?",
            "La personne du premier extrait est-elle la même que celle du second ?",
            "Vérifie les deux enregistrements : ont-ils le même locuteur ?",
            "Au vu de ces deux énoncés, le locuteur est-il identique dans chacun ?",
        ],
        "diff": [
            "Voici deux extraits audio. Sont-ils prononcés par des personnes différentes ?",
            "Écoute les deux enregistrements. Est-ce que ce sont des locuteurs différents ?",
            "Ces deux segments audio proviennent-ils de locuteurs différents ?",
            "Compare les deux voix. S'agit-il de deux personnes différentes ?",
            "Détermine si les deux extraits sont de personnes différentes.",
            "Les deux voix appartiennent-elles à des personnes différentes ?",
            "Voici deux échantillons de voix. Est-ce que ce sont des personnes différentes qui parlent ?",
            "Dis-moi si les deux extraits ont été enregistrés par des locuteurs différents.",
            "Après les avoir écoutés, dirais-tu qu'il s'agit de deux locuteurs distincts ?",
            "La personne du premier extrait est-elle différente de celle du second ?",
            "Vérifie les deux enregistrements : ont-ils des locuteurs différents ?",
            "Au vu de ces deux énoncés, les locuteurs sont-ils distincts ?",
        ],
    },
}
PROMPT_POLARITIES = ("same", "diff")


def index_audio(root: Path, sources) -> dict:
    """Walk the given sources under `root` into {speaker_id: {video_id: [Path, ...]}}.

    Speaker ids are globally unique across VC1/VC2, so datasets are merged into one
    index. `combined.wav` and any non-numeric clip names are skipped.
    """
    index: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for tag, split, sub in sources:
        base = root / tag / split / sub
        if not base.is_dir():
            logger.warning(f"Missing source dir, skipping: {base}")
            continue
        spk_dirs = sorted(base.iterdir())
        n = 0
        for spk_dir in tqdm(spk_dirs, desc=f"index {tag}/{split}", unit="spk"):
            if not spk_dir.is_dir():
                continue
            spk = spk_dir.name
            for vid_dir in spk_dir.iterdir():
                if not vid_dir.is_dir():
                    continue
                for wav in vid_dir.glob("*.wav"):
                    if wav.stem == "combined":
                        continue
                    index[spk][vid_dir.name].append(wav)
                    n += 1
        logger.info(f"  {tag}/{split}: {n} clips")
    return index


def split_speakers_for_dev(index: dict, dev_frac: float, rng: random.Random):
    """Partition speakers into (train_speakers, dev_speakers), speaker-disjoint."""
    speakers = sorted(index.keys())
    rng.shuffle(speakers)
    n_dev = max(1, int(round(len(speakers) * dev_frac)))
    dev = set(speakers[:n_dev])
    train = set(speakers[n_dev:])
    return train, dev


def _duration(path: Path, cache: dict) -> float:
    d = cache.get(path)
    if d is None:
        info = sf.info(str(path))
        d = round(info.frames / info.samplerate, 3)
        cache[path] = d
    return d


def _eligible_speakers(index, speakers):
    """Speakers usable for positive pairs: at least 2 clips total."""
    out = []
    for spk in speakers:
        if sum(len(v) for v in index[spk].values()) >= 2:
            out.append(spk)
    return out


def _sample_positive(index, spk, rng):
    """Two clips of the same speaker, preferring different videos (cross-session)."""
    videos = [v for v, clips in index[spk].items() if clips]
    if len(videos) >= 2:
        va, vb = rng.sample(videos, 2)
        return rng.choice(index[spk][va]), rng.choice(index[spk][vb])
    # single-video speaker: fall back to two distinct utterances of that video
    clips = index[spk][videos[0]]
    if len(clips) >= 2:
        a, b = rng.sample(clips, 2)
        return a, b
    return None


def _sample_negative(index, spk_a, spk_b, rng):
    """One clip from each of two distinct speakers."""
    a = rng.choice([c for v in index[spk_a].values() for c in v])
    b = rng.choice([c for v in index[spk_b].values() for c in v])
    return a, b


def build_pairs(index, speakers, num_pairs, rng):
    """Generate `num_pairs` balanced (label, clipA, clipB) tuples without duplicates."""
    speakers = sorted(speakers)
    pos_speakers = _eligible_speakers(index, speakers)
    if len(speakers) < 2:
        raise ValueError("Need at least 2 speakers to build negative pairs.")
    if not pos_speakers:
        raise ValueError("No speaker has >=2 clips; cannot build positive pairs.")

    seen = set()
    pairs = []
    n_pos = num_pairs // 2
    n_neg = num_pairs - n_pos
    attempts_cap = num_pairs * 50

    def key(a, b):
        return (str(a), str(b)) if str(a) <= str(b) else (str(b), str(a))

    # positives
    got, attempts = 0, 0
    pbar = tqdm(total=n_pos, desc="positive pairs", unit="pair")
    while got < n_pos and attempts < attempts_cap:
        attempts += 1
        spk = rng.choice(pos_speakers)
        sampled = _sample_positive(index, spk, rng)
        if sampled is None:
            continue
        a, b = sampled
        if a == b:
            continue
        k = ("pos",) + key(a, b)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((1, a, b))
        got += 1
        pbar.update(1)
    pbar.close()
    if got < n_pos:
        logger.warning(f"Only generated {got}/{n_pos} positive pairs (limited pool).")

    # negatives
    got, attempts = 0, 0
    pbar = tqdm(total=n_neg, desc="negative pairs", unit="pair")
    while got < n_neg and attempts < attempts_cap:
        attempts += 1
        sa, sb = rng.sample(speakers, 2)
        a, b = _sample_negative(index, sa, sb, rng)
        k = ("neg",) + key(a, b)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((0, a, b))
        got += 1
        pbar.update(1)
    pbar.close()
    if got < n_neg:
        logger.warning(f"Only generated {got}/{n_neg} negative pairs (limited pool).")

    rng.shuffle(pairs)
    return pairs


def speaker_of(path: Path) -> str:
    # .../<speaker_id>/<video_id>/<utt>.wav
    return path.parent.parent.name


def video_of(path: Path) -> str:
    return path.parent.name


# answer-style -> output subdirectory name.
STYLE_SUBDIR = {"short": "yes_no", "sentence": "chat"}


def build_row_spec(label, a, b, rng, dur_cache):
    """Draw the prompt/answer-fact for a pair once (style-independent)."""
    prompt_lang = rng.choice(list(PROMPT_BANK))
    polarity = rng.choice(PROMPT_POLARITIES)
    bank = PROMPT_BANK[prompt_lang]
    prompt = rng.choice(bank[polarity])
    # "same" -> affirm when same speaker; "diff" -> affirm when different.
    affirmative = bool(label) if polarity == "same" else not bool(label)
    # Pick a clause stating the ground-truth fact (for the sentence style).
    clause = rng.choice(bank["same_clause"] if label else bank["diff_clause"])
    answer_suffix = rng.choice(bank["answer_suffix"])
    spk_a, spk_b = speaker_of(a), speaker_of(b)
    da, db = _duration(a, dur_cache), _duration(b, dur_cache)
    return {
        "prompt": prompt, "prompt_lang": prompt_lang, "polarity": polarity,
        "affirmative": affirmative, "clause": clause, "answer_suffix": answer_suffix,
        "bank": bank,
        "label": label, "a": a, "b": b, "spk_a": spk_a, "spk_b": spk_b,
        "da": da, "db": db,
        "uid": f"{spk_a}_{video_of(a)}_{a.stem}__{spk_b}_{video_of(b)}_{b.stem}",
    }


def render_prompt(spec, style):
    # yes/no style appends the explicit answer instruction; chat keeps the bare question.
    if style == "short":
        return f"{spec['prompt']} {spec['answer_suffix']}"
    return spec["prompt"]


def render_answer(spec, style):
    bank = spec["bank"]
    if style == "short":
        return bank["yes"] if spec["affirmative"] else bank["no"]
    lead = bank["yes_lead"] if spec["affirmative"] else bank["no_lead"]
    return f"{lead}, {spec['clause']}"


def make_row(spec, style):
    a, b = spec["a"], spec["b"]
    return NemoDatasetRow(
        id=spec["uid"],
        dataset_name=DATASET_NAME,
        # The row language follows the prompt/answer language, not the audio (which is
        # always English). A French prompt -> French answer -> language="fr".
        language=spec["prompt_lang"],
        turns=[
            NemoTurn(role="User", value=render_prompt(spec, style), turn_type="text"),
            NemoTurn(role="User", value=str(a), turn_type="audio", duration=spec["da"]),
            NemoTurn(role="User", value=str(b), turn_type="audio", duration=spec["db"]),
            NemoTurn(role="Assistant", value=render_answer(spec, style), turn_type="text"),
        ],
        # custom_metadata={
        #     "label": int(spec["label"]),
        #     "same_speaker": bool(spec["label"]),
        #     "answer_style": style,
        #     "prompt_lang": spec["prompt_lang"],
        #     "prompt_polarity": spec["polarity"],
        #     "speaker_a": spec["spk_a"],
        #     "speaker_b": spec["spk_b"],
        #     "video_a": video_of(a),
        #     "video_b": video_of(b),
        #     "audio_a": str(a),
        #     "audio_b": str(b),
        #     "duration_a": spec["da"],
        #     "duration_b": spec["db"],
        # },
    )


def write_split(name, index, speakers, num_pairs, manifest_dir, styles, rng, dur_cache, force):
    targets = {s: manifest_dir / STYLE_SUBDIR[s] / f"{name}.jsonl" for s in styles}
    pending = [s for s, p in targets.items() if force or not p.exists()]
    for s in styles:
        if s not in pending:
            logger.info(f"[{name}] {s} exists, skipping (use --force): {targets[s]}")
    if not pending:
        return
    logger.info(f"[{name}] building {num_pairs} pairs from {len(speakers)} speakers "
                f"(styles: {', '.join(pending)}) ...")
    pairs = build_pairs(index, speakers, num_pairs, rng)
    outs = {s: NemoDataset(name=DATASET_NAME) for s in pending}
    for label, a, b in tqdm(pairs, desc=name):
        spec = build_row_spec(label, a, b, rng, dur_cache)
        for s in pending:
            outs[s].append(make_row(spec, s))
    for s, out in outs.items():
        targets[s].parent.mkdir(parents=True, exist_ok=True)
        out.save(targets[s])
        logger.info(f"[{name}] {s}: wrote {len(out)} rows -> {targets[s]}")


def main():
    parser = argparse.ArgumentParser(description="Convert VoxCeleb1/2 to NeMo speaker-verification manifests")
    parser.add_argument("--root", type=str,
                        default=f"{os.environ['DATA_FOLDER']}/raw/misc/en",
                        help="Dir containing VoxCeleb1/ and VoxCeleb2/.")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Output dir for {train,dev,test}.jsonl.")
    parser.add_argument("--num-pairs-train", type=int, default=200000)
    parser.add_argument("--num-pairs-dev", type=int, default=5000)
    parser.add_argument("--num-pairs-test", type=int, default=20000)
    parser.add_argument("--dev-speaker-frac", type=float, default=0.05,
                        help="Fraction of train_dev speakers held out (speaker-disjoint) for dev.")
    parser.add_argument("--answer-style", choices=["short", "sentence", "both"], default="both",
                        help="'short' = yes/no (subdir short/); 'sentence' = full-sentence chat "
                             "answer (subdir chat/); 'both' = render both on the same pairs.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    styles = ["short", "sentence"] if args.answer_style == "both" else [args.answer_style]

    root = Path(args.root)
    if args.manifest_path is None:
        args.manifest_path = f"{os.environ['DATA_FOLDER']}/nemo/misc/multilang/context/VoxCeleb"
    manifest_dir = Path(args.manifest_path)
    rng = random.Random(args.seed)
    dur_cache: dict = {}

    # train + dev (speaker-disjoint)
    td_index = index_audio(root, SOURCES["train_dev"])
    train_spk, dev_spk = split_speakers_for_dev(td_index, args.dev_speaker_frac, rng)
    logger.info(f"train speakers: {len(train_spk)}  dev speakers: {len(dev_spk)}")
    write_split("train", td_index, train_spk, args.num_pairs_train,
                manifest_dir, styles, rng, dur_cache, args.force)
    write_split("dev", td_index, dev_spk, args.num_pairs_dev,
                manifest_dir, styles, rng, dur_cache, args.force)

    # test
    test_index = index_audio(root, SOURCES["test"])
    write_split("test", test_index, set(test_index.keys()), args.num_pairs_test,
                manifest_dir, styles, rng, dur_cache, args.force)


if __name__ == "__main__":
    main()
