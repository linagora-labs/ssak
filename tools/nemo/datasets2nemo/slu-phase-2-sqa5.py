import argparse
import logging
import re
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("/data-server/datasets/audio/raw/misc/en/context/slu-phase-2-sqa5")
MANIFEST_PATH_AUDIOQ_TEXTC = Path("/data-server/datasets/audio/nemo/question-answering/qa_audio-question-text-context/en/slu-phase-2-sqa5")
MANIFEST_PATH_AUDIOQ_AUDIOC = Path("/data-server/datasets/audio/nemo/question-answering/qa_audio-question-audio-context/en/slu-phase-2-sqa5")
MANIFEST_PATH_TEXTQ_AUDIOC = Path("/data-server/datasets/audio/nemo/question-answering/qa_audio-context-text-question/en/slu-phase-2-sqa5")
AUDIO_PATH = Path("/data-server/datasets/audio/raw/misc/en/slu-phase-2-sqa5/audios")
DATASET_NAME = "slue-phase-2 - SQA5"
SPLITS = ["train", "validation", "test", "verified_test"]


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def write_flac_if_missing(path: Path, hf_audio: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(hf_audio["array"], dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0]
    sr = int(hf_audio["sampling_rate"])
    if not path.exists():
        sf.write(path, arr, sr)
    return {"path": path, "duration": len(arr) / sr}


def recover_cased_answer(answer_spans: dict, word2time: dict) -> str:
    """Recover original casing of the answer by locating it in word2time.

    word2time uses start_second == end_second == -1 for punctuation tokens,
    which are skipped when matching the answer span boundaries.
    """
    answer = answer_spans["answer"][0]
    try:
        words  = list(word2time["word"])
        starts = list(word2time["start_second"])
        ends   = list(word2time["end_second"])
        a_start = answer_spans["start_second"][0]
        a_end   = answer_spans["end_second"][0]
        i0 = next(i for i, s in enumerate(starts) if s != -1 and s == a_start)
        i1 = next(i for i, e in enumerate(ends)   if e != -1 and e == a_end)
        cased = " ".join(
            w for w, s in zip(words[i0:i1 + 1], starts[i0:i1 + 1]) if s != -1
        )
        norm = lambda s: re.sub(r"[^\w]", "", s).lower()
        if cased and norm(cased) == norm(answer):
            return cased
        logger.warning(f"Casing recovery mismatch: {cased!r} vs {answer!r}")
    except (StopIteration, KeyError, IndexError):
        logger.warning(f"Casing recovery failed for answer {answer!r}")
    return answer


def load_source_dataset(cache_dir: str | None, from_disk: str | None):
    if from_disk:
        logger.info(f"Loading dataset from disk: {from_disk}")
        return datasets.load_from_disk(from_disk)
    logger.info(f"Loading asapp/slue-phase-2 (sqa5)"
                + (f" from cache_dir={cache_dir}" if cache_dir else ""))
    return datasets.load_dataset("asapp/slue-phase-2", "sqa5", cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert slue-phase-2 sqa5 to NeMo manifest")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF datasets cache dir (reuses files already downloaded via load_dataset).")
    parser.add_argument("--from-disk", type=str, default=None,
                        help="Path to a dataset saved via save_to_disk (takes precedence over --cache-dir).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    args = parser.parse_args()

    AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    for p in (MANIFEST_PATH, MANIFEST_PATH_AUDIOQ_TEXTC, MANIFEST_PATH_AUDIOQ_AUDIOC, MANIFEST_PATH_TEXTQ_AUDIOC):
        p.mkdir(parents=True, exist_ok=True)

    ds = load_source_dataset(args.cache_dir, args.from_disk)

    for split in SPLITS:
        if split not in ds:
            logger.info(f"Split {split!r} not in dataset, skipping")
            continue

        targets = {
            "full":           MANIFEST_PATH               / f"{split}.jsonl",
            "audioQ_textC":   MANIFEST_PATH_AUDIOQ_TEXTC  / f"{split}.jsonl",
            "audioQ_audioC":  MANIFEST_PATH_AUDIOQ_AUDIOC / f"{split}.jsonl",
            "textQ_audioC":   MANIFEST_PATH_TEXTQ_AUDIOC  / f"{split}.jsonl",
        }
        pending = {k: v for k, v in targets.items() if args.force or not v.exists()}
        if not pending:
            logger.info(f"[{split}] all manifests already exist, skipping")
            continue
        for k, v in targets.items():
            if k not in pending:
                logger.info(f"[{split}] {k} already exists, will skip save: {v}")

        subset = ds[split]
        logger.info(f"[{split}] total={len(subset)}")

        outs = {k: NemoDataset(name=DATASET_NAME) for k in pending}

        for row in tqdm(subset, desc=split):
            uid = f"{row['question_id']}-{row['question_speaker_id']}-{row['document_speaker_id']}"

            doc_audio = write_flac_if_missing(
                AUDIO_PATH / "documents" / f"{sanitize(row['document_id'])}.flac",
                row["document_audio"],
            )
            q_audio = write_flac_if_missing(
                AUDIO_PATH / "questions" / f"{sanitize(uid)}.flac",
                row["question_audio"],
            )
            answer = recover_cased_answer(row["answer_spans"], row.get("word2time") or {})

            doc_audio_turn = NemoTurn(role="User", value=str(doc_audio["path"]), turn_type="audio",
                                      duration=round(doc_audio["duration"], 3))
            q_audio_turn   = NemoTurn(role="User", value=str(q_audio["path"]),   turn_type="audio",
                                      duration=round(q_audio["duration"], 3))
            q_text_turn    = NemoTurn(role="User",      value=row["raw_question_text"], turn_type="text")
            answer_turn    = NemoTurn(role="Assistant", value=answer,                    turn_type="text")

            if "full" in outs:
                outs["full"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[q_text_turn, doc_audio_turn, answer_turn],
                    custom_metadata={
                        "question_id":              row["question_id"],
                        "document_id":              row["document_id"],
                        "question-source":          row["question_id"].split("-")[0],
                        "question_speaker_id":      row["question_speaker_id"],
                        "document_speaker_id":      row["document_speaker_id"],
                        "raw_question_text":        row["raw_question_text"],
                        "normalized_question_text": row.get("normalized_question_text"),
                        "raw_document_text":        row.get("raw_document_text"),
                        "normalized_document_text": row.get("normalized_document_text"),
                        "question_audio_filepath":  str(q_audio["path"]),
                        "question_audio_duration":  round(q_audio["duration"], 3),
                        "answer_spans":             {k: list(v) for k, v in row["answer_spans"].items()},
                        "word2time":                {k: list(v) for k, v in (row.get("word2time") or {}).items()},
                    },
                ))

            if "audioQ_textC" in outs:
                outs["audioQ_textC"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[
                        q_audio_turn,
                        NemoTurn(role="User", value=row.get("raw_document_text"), turn_type="text"),
                        answer_turn,
                    ],
                ))

            if "audioQ_audioC" in outs:
                outs["audioQ_audioC"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[q_audio_turn, doc_audio_turn, answer_turn],
                ))

            if "textQ_audioC" in outs:
                outs["textQ_audioC"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[q_text_turn, doc_audio_turn, answer_turn],
                ))

        for key, out in outs.items():
            manifest_file = targets[key]
            out.save(manifest_file)
            logger.info(f"[{split}] {key}: wrote {len(out)} rows → {manifest_file}")


if __name__ == "__main__":
    main()
