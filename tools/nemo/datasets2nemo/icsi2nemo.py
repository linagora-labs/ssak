"""Convert the ICSI Meeting Corpus (NXT-format) into NeMo manifests.

Expected layout:

    <root>/
        Signals/<MEETING>/<MEETING>.interaction.wav
        annotations/ICSIplus/
            Words/<MEETING>.<SPEAKER>.words.xml
            Segments/<MEETING>.<SPEAKER>.segs.xml
            DialogueActs/<MEETING>.<SPEAKER>.dialogue-acts.xml
            Contributions/
                Summarization/abstractive/<MEETING>.abssumm.xml
                Summarization/extractive/<MEETING>.summlink.xml
                TopicSegmentation/<MEETING>.topic.xml

Produces three manifest families (each opt-in via a flag):
  - summary/      : one row per meeting (interaction.wav + abstractive summary text)
  - asr/          : one row per segment (interaction.wav with offset + transcript)
  - diarization/  : windowed "who spoke when" rows in asr/timestamps/timestamps_asr
                    variants, several output formats (RTTM, SRT, VTT, ...) and
                    with/without backchannels (written under a diarization-specific root)
"""

import argparse
import logging
import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn
from prompts_diarization import (
    DIAR_VARIANTS,
    _DIAR_FORMATS,
    choose_format,
    clean_window_pieces,
    make_diar_lean_row,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "ICSI"

NITE_NS = "{http://nite.sourceforge.net/}"

# Standard split used in ICSI summarization papers (Shang et al. 2018, etc.).
DEV_MEETINGS = {
    "Bmr023", "Bro018",
}
TEST_MEETINGS = {
    "Bed004", "Bed009", "Bed014", "Bed016",
    "Bmr005", "Bmr019",
}


def split_for(meeting_id: str) -> str:
    if meeting_id in DEV_MEETINGS:
        return "validation"
    if meeting_id in TEST_MEETINGS:
        return "test"
    return "train"


def audio_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        logger.warning(f"Could not read duration for {path}: {e}")
        return None


# --------------------------- abstractive summary ---------------------------

_ABS_SECTION_ALIASES = {
    "abstract": "abstract",
    "decisions": "decisions",
    "problems": "problems",
    "progress": "progress",
}

_SECTION_PROMPTS = {
    "abstract": [
        "Summarize this meeting.",
        "Provide a summary of what was discussed.",
        "Give an overview of this part of the conversation.",
        "What are the main points covered in this recording?",
        "Give me a summary of the audio.",
    ],
    "decisions": [
        "What decisions were made in this meeting?",
        "List the decisions that were reached during this discussion.",
        "Summarize the key decisions from this segment.",
        "What did the participants decide on?",
    ],
    "problems": [
        "What problems were raised in this clip?",
        "Identify the issues discussed in this audio.",
        "What challenges or concerns were brought up?",
        "Summarize the problems mentioned during this discussion.",
    ],
    "progress": [
        "What progress was reported in this meeting?",
        "Summarize the progress discussed in this audio.",
        "What advancements or updates were mentioned?",
        "Describe the progress made as discussed in this meeting segment.",
    ],
}


def parse_abstractive_summary(xml_path: Path) -> dict:
    """Return {'abstract': str, 'decisions': str, 'problems': str, 'progress': str}."""
    sentences = parse_abstractive_sentences(xml_path)
    buckets: dict[str, list[str]] = {"abstract": [], "decisions": [], "problems": [], "progress": []}
    for sent in sentences.values():
        buckets[sent["section"]].append(sent["text"])
    return {k: " ".join(v).strip() for k, v in buckets.items()}


def parse_abstractive_sentences(xml_path: Path) -> dict:
    """Return {sentence_id: {'text': str, 'section': str}} preserving XML order."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: dict[str, dict] = {}
    for section in root:
        tag = section.tag.split("}")[-1].lower()
        bucket = _ABS_SECTION_ALIASES.get(tag)
        if bucket is None:
            continue
        for sent in section.iter():
            if sent.tag.split("}")[-1].lower() != "sentence":
                continue
            sid = sent.get(f"{NITE_NS}id") or sent.get("id")
            text = (sent.text or "").strip()
            if sid and text:
                out[sid] = {"text": text, "section": bucket}
    return out


# --------------------------- words + segments ---------------------------

WORDS_FILE_RE = re.compile(r"^(?P<meeting>[^.]+)\.(?P<speaker>[^.]+)\.words\.xml$")
SEG_FILE_RE = re.compile(r"^(?P<meeting>[^.]+)\.(?P<speaker>[^.]+)\.segs\.xml$")
SEG_HREF_RE = re.compile(
    r"(?P<file>[^#]+)#id\((?P<start>[^)]+)\)(?:\.\.id\((?P<end>[^)]+)\))?"
)


def load_words(xml_path: Path) -> dict:
    """Parse a words.xml file into {nite_id: {'text': str|None, 'start': float|None, 'end': float|None}}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    words = {}
    for elem in root:
        tag = elem.tag.split("}")[-1]
        nite_id = elem.get(f"{NITE_NS}id") or elem.get("id")
        if nite_id is None:
            continue
        start = elem.get("starttime")
        end = elem.get("endtime")
        try:
            start_f = float(start) if start is not None else None
        except ValueError:
            start_f = None
        try:
            end_f = float(end) if end is not None else None
        except ValueError:
            end_f = None
        text = elem.text.strip() if (tag == "w" and elem.text) else None
        words[nite_id] = {"text": text, "start": start_f, "end": end_f}
    return words


def load_segments(xml_path: Path) -> dict:
    """Parse a segs.xml file into {segment_id: {'start': float, 'end': float, 'word_refs': [...]}}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    segments = {}
    for seg in root:
        if seg.tag.split("}")[-1] != "segment":
            continue
        seg_id = seg.get(f"{NITE_NS}id") or seg.get("id")
        try:
            start = float(seg.get("starttime")) if seg.get("starttime") else None
            end = float(seg.get("endtime")) if seg.get("endtime") else None
        except ValueError:
            start = end = None
        word_refs = []
        for child in seg.iter(f"{NITE_NS}child"):
            href = child.get("href")
            if not href:
                continue
            m = SEG_HREF_RE.search(href)
            if m:
                word_refs.append((m.group("file"), m.group("start"), m.group("end")))
        segments[seg_id] = {"start": start, "end": end, "word_refs": word_refs}
    return segments


def resolve_segment_words(word_refs, words_by_file: dict):
    """Expand (file, start_id, end_id) ranges into a flat list of word dicts."""
    out = []
    for file_, start_id, end_id in word_refs:
        words = words_by_file.get(file_)
        if not words:
            continue
        ids = list(words.keys())
        if start_id not in words:
            continue
        i0 = ids.index(start_id)
        i1 = ids.index(end_id) if (end_id and end_id in words) else i0
        for nid in ids[i0:i1 + 1]:
            out.append(words[nid])
    return out


# NXT stores punctuation as standalone word tokens (<w punc="true">), so a naive
# space-join yields "Okay ." / "Hi , I'm". Drop the space before such punctuation.
_DETOK_RE = re.compile(r"\s+([,.;:!?])")


def detokenize(text: str) -> str:
    return _DETOK_RE.sub(r"\1", text).strip()


def segment_text(word_records: list) -> str:
    parts = [w["text"] for w in word_records if w.get("text")]
    return detokenize(" ".join(parts))


# --------------------------- topics + dialogue acts + summlink ---------------------------

def parse_topics(xml_path: Path) -> list[dict]:
    """Return flat list of leaf topics with segment references.

    ICSI topics can be nested; we flatten them, collecting segment refs
    from each <topic> that has no sub-topics (leaf) or treating each
    topic level independently with its direct segment children.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    topics = []
    _collect_topics(root, topics)
    return topics


def _collect_topics(element, out: list):
    """Recursively collect topics. Each topic becomes one entry with its direct segment refs."""
    for child in element:
        if child.tag.split("}")[-1].lower() != "topic":
            continue
        tid = child.get(f"{NITE_NS}id") or child.get("id")
        desc = child.get("description") or ""
        seg_refs = []
        for nite_child in child.iter(f"{NITE_NS}child"):
            href = nite_child.get("href")
            if not href:
                continue
            m = SEG_HREF_RE.search(href)
            if m and m.group("file").endswith(".segs.xml"):
                seg_refs.append((m.group("file"), m.group("start"), m.group("end")))
        has_subtopics = any(
            c.tag.split("}")[-1].lower() == "topic" for c in child
        )
        if not has_subtopics and seg_refs:
            out.append({"id": tid, "description": desc, "seg_refs": seg_refs})
        _collect_topics(child, out)


def seg_ref_time_span(seg_refs, segments_by_file: dict) -> tuple[float | None, float | None]:
    """Return (min_start, max_end) over all segments referenced."""
    starts, ends = [], []
    for file_, start_id, end_id in seg_refs:
        segs = segments_by_file.get(file_)
        if not segs:
            continue
        ids = list(segs.keys())
        if start_id not in segs:
            continue
        i0 = ids.index(start_id)
        i1 = ids.index(end_id) if (end_id and end_id in segs) else i0
        for sid in ids[i0:i1 + 1]:
            seg = segs[sid]
            if seg.get("start") is not None:
                starts.append(seg["start"])
            if seg.get("end") is not None:
                ends.append(seg["end"])
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def parse_dialog_acts(dialog_acts_dir: Path, meeting_id: str) -> dict:
    """Return {da_id: [(file, start_id, end_id), ...]} across all speakers."""
    out: dict[str, list] = {}
    for fp in dialog_acts_dir.glob(f"{meeting_id}.*.dialogue-acts.xml"):
        try:
            root = ET.parse(fp).getroot()
        except ET.ParseError as e:
            logger.warning(f"{fp}: bad XML ({e})")
            continue
        for dact in root:
            if dact.tag.split("}")[-1].lower() != "dialogueact":
                continue
            did = dact.get(f"{NITE_NS}id") or dact.get("id")
            if not did:
                continue
            refs = []
            for child in dact.iter(f"{NITE_NS}child"):
                href = child.get("href")
                if not href:
                    continue
                m = SEG_HREF_RE.search(href)
                if m and m.group("file").endswith(".words.xml"):
                    refs.append((m.group("file"), m.group("start"), m.group("end")))
            if refs:
                out[did] = refs
        for dact in root:
            if dact.tag.split("}")[-1].lower() != "dialogueact":
                continue
            did = dact.get(f"{NITE_NS}id") or dact.get("id")
            if not did or did in out:
                continue
            start = dact.get("starttime")
            end = dact.get("endtime")
            if start is not None and end is not None:
                try:
                    out[did] = [("__timing__", float(start), float(end))]
                except ValueError:
                    pass
    return out


_LINK_RE = re.compile(r"(?P<file>[^#]+)#id\((?P<id>[^)]+)\)")


def parse_summlinks(xml_path: Path) -> list[tuple[str, str]]:
    """Return list of (abstractive_sentence_id, dialog_act_id) pairs."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pairs = []
    for link in root:
        tag = link.tag.split("}")[-1].lower()
        if tag not in ("summlink", "summarylink"):
            continue
        abs_id = da_id = None
        for child in link:
            ctag = child.tag.split("}")[-1].lower()
            if ctag not in ("child", "pointer"):
                continue
            href = child.get("href")
            if not href:
                continue
            m = _LINK_RE.search(href)
            if not m:
                continue
            file_ = m.group("file")
            target = m.group("id")
            if "abssumm" in file_:
                abs_id = target
            elif "dialogue-act" in file_:
                da_id = target
        if abs_id and da_id:
            pairs.append((abs_id, da_id))
    return pairs


def word_ref_time_span(word_refs, words_by_file):
    """Return (min_start, max_end) over all words referenced."""
    starts, ends = [], []
    for file_, start_id, end_id in word_refs:
        if file_ == "__timing__":
            starts.append(start_id)
            ends.append(end_id)
            continue
        words = words_by_file.get(file_)
        if not words:
            continue
        ids = list(words.keys())
        if start_id not in words:
            continue
        i0 = ids.index(start_id)
        i1 = ids.index(end_id) if (end_id and end_id in words) else i0
        for nid in ids[i0:i1 + 1]:
            w = words[nid]
            if w.get("start") is not None:
                starts.append(w["start"])
            if w.get("end") is not None:
                ends.append(w["end"])
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def assign_sentences_to_topics(topic_spans, sentences, summlinks, da_refs, words_by_file):
    """Return {topic_index: [sentence_id, ...]}.

    Each sentence is assigned to the topic with the largest temporal overlap
    of its linked dialogue-act span.
    """
    sent_to_das: dict[str, list[str]] = {}
    for sid, did in summlinks:
        sent_to_das.setdefault(sid, []).append(did)

    assignment: dict[int, list[str]] = {i: [] for i in range(len(topic_spans))}
    for sid in sentences:
        das = sent_to_das.get(sid, [])
        if not das:
            continue
        starts, ends = [], []
        for did in das:
            refs = da_refs.get(did)
            if not refs:
                continue
            s, e = word_ref_time_span(refs, words_by_file)
            if s is not None and e is not None:
                starts.append(s)
                ends.append(e)
        if not starts:
            continue
        sent_start, sent_end = min(starts), max(ends)

        best_i, best_overlap = None, 0.0
        for i, (t_start, t_end) in enumerate(topic_spans):
            if t_start is None or t_end is None:
                continue
            overlap = max(0.0, min(sent_end, t_end) - max(sent_start, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_i = i
        if best_i is not None:
            assignment[best_i].append(sid)
    return assignment


# --------------------------- per-meeting helpers ---------------------------

def _load_meeting_words(words_dir: Path, meeting_id: str) -> dict:
    words_by_file = {}
    for fp in words_dir.glob(f"{meeting_id}.*.words.xml"):
        try:
            words_by_file[fp.name] = load_words(fp)
        except ET.ParseError as e:
            logger.warning(f"{fp}: {e}")
    return words_by_file


def _load_meeting_segments(seg_dir: Path, meeting_id: str) -> dict:
    segments_by_file = {}
    for fp in seg_dir.glob(f"{meeting_id}.*.segs.xml"):
        try:
            segments_by_file[fp.name] = load_segments(fp)
        except ET.ParseError as e:
            logger.warning(f"{fp}: {e}")
    return segments_by_file


def discover_meetings(signals_dir: Path) -> list[str]:
    return sorted([p.name for p in signals_dir.iterdir() if p.is_dir()])


def collect_speaker_files(words_dir: Path, meeting_id: str) -> dict:
    """Return {speaker_letter: words_xml_path} for this meeting."""
    speakers = {}
    for fp in words_dir.glob(f"{meeting_id}.*.words.xml"):
        m = WORDS_FILE_RE.match(fp.name)
        if m:
            speakers[m.group("speaker")] = fp
    return speakers


def build_summary_row(meeting_id: str, mix_audio: Path, summary: dict,
                      duration: float) -> NemoDatasetRow | None:
    main = summary.get("abstract", "").strip()
    if not main:
        return None
    audio_turn = NemoTurn(role="User", value=str(mix_audio), turn_type="audio",
                          duration=round(duration, 3))
    summary_turn = NemoTurn(role="Assistant", value=main, turn_type="text")
    return NemoDatasetRow(
        id=meeting_id,
        dataset_name=DATASET_NAME,
        split=split_for(meeting_id),
        language="en",
        turns=[audio_turn, summary_turn],
        custom_metadata={
            "meeting":   meeting_id,
            "abstract":  main,
            "decisions": summary.get("decisions") or None,
            "problems":  summary.get("problems") or None,
            "progress":  summary.get("progress") or None,
        },
    )


def build_section_summary_rows(meeting_id: str, mix_audio: Path, summary: dict,
                                duration: float) -> list[NemoDatasetRow]:
    """Emit one row per non-empty section (abstract, decisions, problems, progress)."""
    rows = []
    for section, text in summary.items():
        text = text.strip()
        if not text:
            continue
        audio_turn = NemoTurn(role="User", value=str(mix_audio), turn_type="audio",
                              duration=round(duration, 3))
        summary_turn = NemoTurn(role="Assistant", value=text, turn_type="text")
        rows.append(NemoDatasetRow(
            id=f"{meeting_id}.{section}",
            dataset_name=DATASET_NAME,
            split=split_for(meeting_id),
            language="en",
            turns=[audio_turn, summary_turn],
            custom_metadata={
                "meeting": meeting_id,
                "section": section,
            },
        ))
    return rows


def build_topic_summary_rows(meeting_id: str, mix_audio: Path,
                              words_dir: Path, seg_dir: Path,
                              dialog_acts_dir: Path, abs_dir: Path,
                              extractive_dir: Path, topics_dir: Path,
                              mix_duration: float, min_dur: float, max_dur: float,
                              per_section: bool = False):
    """Emit one summary row per topic with offset/duration on mix_audio."""
    topics_xml = topics_dir / f"{meeting_id}.topic.xml"
    abs_xml = abs_dir / f"{meeting_id}.abssumm.xml"
    summlink_xml = extractive_dir / f"{meeting_id}.summlink.xml"
    if not (topics_xml.exists() and abs_xml.exists() and summlink_xml.exists()):
        logger.debug(f"{meeting_id}: missing topics/abssumm/summlink, skipping topic-segmented summary")
        return []

    words_by_file = _load_meeting_words(words_dir, meeting_id)
    segments_by_file = _load_meeting_segments(seg_dir, meeting_id)

    try:
        topics = parse_topics(topics_xml)
        sentences = parse_abstractive_sentences(abs_xml)
        summlinks = parse_summlinks(summlink_xml)
    except ET.ParseError as e:
        logger.warning(f"{meeting_id}: XML parse error ({e})")
        return []

    da_refs = parse_dialog_acts(dialog_acts_dir, meeting_id)

    topic_spans = [seg_ref_time_span(t["seg_refs"], segments_by_file) for t in topics]
    assignment = assign_sentences_to_topics(topic_spans, sentences, summlinks, da_refs, words_by_file)

    rows = []
    for i, topic in enumerate(topics):
        t_start, t_end = topic_spans[i]
        if t_start is None or t_end is None:
            continue
        t_end = min(t_end, mix_duration)
        duration = t_end - t_start
        if duration < min_dur or duration > max_dur:
            logger.debug(f"{meeting_id}/{topic['id']}: duration {duration:.1f}s outside [{min_dur},{max_dur}], skipping")
            continue
        sent_ids = assignment.get(i, [])
        if not sent_ids:
            continue

        if per_section:
            by_section: dict[str, list[str]] = {}
            for sid in sent_ids:
                by_section.setdefault(sentences[sid]["section"], []).append(sid)
            for section, sids in by_section.items():
                ordered = [sid for sid in sentences if sid in set(sids)]
                text = " ".join(sentences[sid]["text"] for sid in ordered).strip()
                if not text:
                    continue
                audio_turn = NemoTurn(
                    role="User", value=str(mix_audio), turn_type="audio",
                    duration=round(duration, 3), offset=round(t_start, 3),
                )
                summary_turn = NemoTurn(role="Assistant", value=text, turn_type="text")
                rows.append(NemoDatasetRow(
                    id=f"{topic['id']}.{section}",
                    dataset_name=DATASET_NAME,
                    split=split_for(meeting_id),
                    language="en",
                    turns=[audio_turn, summary_turn],
                    custom_metadata={
                        "meeting":     meeting_id,
                        "topic_id":    topic["id"],
                        "section":     section,
                        "description": topic["description"] or None,
                        "n_sentences": len(ordered),
                    },
                ))
        else:
            ordered = [sid for sid in sentences if sid in set(sent_ids)]
            text = " ".join(sentences[sid]["text"] for sid in ordered).strip()
            if not text:
                continue
            audio_turn = NemoTurn(
                role="User", value=str(mix_audio), turn_type="audio",
                duration=round(duration, 3), offset=round(t_start, 3),
            )
            summary_turn = NemoTurn(role="Assistant", value=text, turn_type="text")
            rows.append(NemoDatasetRow(
                id=f"{topic['id']}",
                dataset_name=DATASET_NAME,
                split=split_for(meeting_id),
                language="en",
                turns=[audio_turn, summary_turn],
                custom_metadata={
                    "meeting":     meeting_id,
                    "topic_id":    topic["id"],
                    "description": topic["description"] or None,
                    "n_sentences": len(ordered),
                },
            ))
    return rows


def build_turn_summary_rows(meeting_id: str, mix_audio: Path,
                             words_dir: Path, dialog_acts_dir: Path,
                             abs_dir: Path, extractive_dir: Path,
                             mix_duration: float, min_dur: float, max_dur: float):
    """Emit one row per abstractive sentence, audio span from linked dialogue acts."""
    abs_xml = abs_dir / f"{meeting_id}.abssumm.xml"
    summlink_xml = extractive_dir / f"{meeting_id}.summlink.xml"
    if not (abs_xml.exists() and summlink_xml.exists()):
        logger.debug(f"{meeting_id}: missing abssumm/summlink, skipping turn-level summary")
        return []

    words_by_file = _load_meeting_words(words_dir, meeting_id)

    try:
        sentences = parse_abstractive_sentences(abs_xml)
        summlinks = parse_summlinks(summlink_xml)
    except ET.ParseError as e:
        logger.warning(f"{meeting_id}: XML parse error ({e})")
        return []

    da_refs = parse_dialog_acts(dialog_acts_dir, meeting_id)

    sent_to_das: dict[str, list[str]] = {}
    for sid, did in summlinks:
        sent_to_das.setdefault(sid, []).append(did)

    rows = []
    for sid, sent in sentences.items():
        das = sent_to_das.get(sid, [])
        if not das:
            continue
        starts, ends = [], []
        for did in das:
            refs = da_refs.get(did)
            if not refs:
                continue
            s, e = word_ref_time_span(refs, words_by_file)
            if s is not None and e is not None:
                starts.append(s)
                ends.append(e)
        if not starts:
            continue
        t_start = min(starts)
        t_end = min(max(ends), mix_duration)
        duration = t_end - t_start
        if duration <= 0 or duration < min_dur or duration > max_dur:
            continue

        text = sent["text"].strip()
        if not text:
            continue

        audio_turn = NemoTurn(
            role="User", value=str(mix_audio), turn_type="audio",
            duration=round(duration, 3), offset=round(t_start, 3),
        )
        summary_turn = NemoTurn(role="Assistant", value=text, turn_type="text")
        rows.append(NemoDatasetRow(
            id=sid,
            dataset_name=DATASET_NAME,
            split=split_for(meeting_id),
            language="en",
            turns=[audio_turn, summary_turn],
            custom_metadata={
                "meeting":        meeting_id,
                "sentence_id":    sid,
                "section":        sent["section"],
                "n_dialogue_acts": len(das),
            },
        ))
    return rows


def build_asr_rows(meeting_id: str, mix_audio: Path, words_dir: Path,
                   seg_dir: Path, mix_duration: float,
                   min_duration: float, max_duration: float):
    rows = []
    speakers = collect_speaker_files(words_dir, meeting_id)
    if not speakers:
        return rows

    words_by_file = {}
    for letter, words_xml in speakers.items():
        try:
            words_by_file[words_xml.name] = load_words(words_xml)
        except ET.ParseError as e:
            logger.warning(f"{meeting_id}/{letter}: bad words.xml ({e}), skipping speaker")

    for letter in sorted(speakers):
        seg_xml = seg_dir / f"{meeting_id}.{letter}.segs.xml"
        if not seg_xml.exists():
            continue

        try:
            segs = load_segments(seg_xml)
        except ET.ParseError as e:
            logger.warning(f"{meeting_id}/{letter}: bad segs.xml ({e}), skipping")
            continue

        for seg_id, seg_data in segs.items():
            start = seg_data["start"]
            end = seg_data["end"]
            if start is None or end is None:
                continue

            word_records = resolve_segment_words(seg_data["word_refs"], words_by_file)
            text = segment_text(word_records)
            if not text:
                continue

            end = min(end, mix_duration)
            duration = end - start
            if duration <= 0 or duration < min_duration or duration > max_duration:
                continue

            audio_turn = NemoTurn(
                role="User", value=str(mix_audio), turn_type="audio",
                duration=round(duration, 3), offset=round(start, 3),
            )
            text_turn = NemoTurn(role="Assistant", value=text, turn_type="text")
            rows.append(NemoDatasetRow(
                id=seg_id or f"{meeting_id}.{letter}.{len(rows)}",
                dataset_name=DATASET_NAME,
                split=split_for(meeting_id),
                language="en",
                speaker=f"{meeting_id}.{letter}",
                turns=[audio_turn, text_turn],
                custom_metadata={
                    "meeting":        meeting_id,
                    "speaker_letter": letter,
                },
            ))
    return rows


# --------------------------- diarization ---------------------------

def collect_meeting_turns(meeting_id: str, words_dir: Path, seg_dir: Path,
                          merge_gap: float | None, mix_duration: float | None = None) -> list[dict]:
    """Return chronologically-ordered speaker turns for the whole meeting.

    Each turn is {'speaker': letter, 'start', 'end', 'text', 'words': [...]},
    where 'words' keeps per-word timing so a too-long turn can be split at word
    boundaries. Consecutive turns of the same speaker separated by <= merge_gap
    seconds are merged (merge_gap=None disables merging)."""
    speakers = collect_speaker_files(words_dir, meeting_id)
    if not speakers:
        return []

    words_by_file = {}
    for letter, words_xml in speakers.items():
        try:
            words_by_file[words_xml.name] = load_words(words_xml)
        except ET.ParseError as e:
            logger.warning(f"{meeting_id}/{letter}: bad words.xml ({e}), skipping speaker")

    units: list[dict] = []
    for letter in sorted(speakers):
        seg_xml = seg_dir / f"{meeting_id}.{letter}.segs.xml"
        if not seg_xml.exists():
            continue
        try:
            segs = load_segments(seg_xml)
        except ET.ParseError as e:
            logger.warning(f"{meeting_id}/{letter}: bad segs.xml ({e}), skipping")
            continue
        for seg_id, seg_data in segs.items():
            word_records = resolve_segment_words(seg_data["word_refs"], words_by_file)
            text = segment_text(word_records)
            if not text:
                continue
            start, end = seg_data["start"], seg_data["end"]
            if start is None or end is None:
                word_starts = [w["start"] for w in word_records if w.get("start") is not None]
                word_ends = [w["end"] for w in word_records if w.get("end") is not None]
                if not word_starts or not word_ends:
                    continue
                start = min(word_starts) if start is None else start
                end = max(word_ends) if end is None else end
            if mix_duration is not None:
                end = min(end, mix_duration)
            if end <= start:
                continue
            words = [
                {"text": w["text"], "start": w["start"], "end": w["end"]}
                for w in word_records if w.get("text")
            ]
            units.append({"speaker": letter, "start": start, "end": end,
                          "text": text, "words": words})

    units.sort(key=lambda u: (u["start"], u["end"]))
    if merge_gap is not None:
        units = _merge_same_speaker(units, merge_gap)
    return units


def _merge_same_speaker(units: list[dict], gap: float) -> list[dict]:
    """Merge consecutive same-speaker turns within `gap` seconds of each other."""
    out: list[dict] = []
    for u in units:
        prev = out[-1] if out else None
        if prev and prev["speaker"] == u["speaker"] and u["start"] - prev["end"] <= gap:
            prev["end"] = max(prev["end"], u["end"])
            prev["text"] = detokenize(prev["text"] + " " + u["text"])
            prev["words"].extend(u["words"])
        else:
            out.append(dict(u, words=list(u["words"])))
    return out


def _mk_piece(speaker: str, words: list[dict]) -> dict:
    text = detokenize(" ".join(w["text"] for w in words if w.get("text")))
    return {"speaker": speaker, "start": words[0]["start"],
            "end": words[-1]["end"], "text": text, "words": list(words)}


def _split_long_turn(unit: dict, max_dur: float) -> list[dict]:
    """Split a turn longer than max_dur at word boundaries; otherwise return [unit]."""
    if unit["end"] - unit["start"] <= max_dur:
        return [unit]
    words = [w for w in unit["words"]
             if w.get("start") is not None and w.get("end") is not None and w.get("text")]
    if not words:
        # No word timing to split on: hard-cut by time, keep text on the first piece.
        pieces, s = [], unit["start"]
        while s < unit["end"]:
            e = min(s + max_dur, unit["end"])
            pieces.append({"speaker": unit["speaker"], "start": s, "end": e,
                           "text": unit["text"] if s == unit["start"] else "", "words": []})
            s = e
        return pieces
    pieces, cur, cur_start = [], [], words[0]["start"]
    for w in words:
        if cur and w["end"] - cur_start > max_dur:
            pieces.append(_mk_piece(unit["speaker"], cur))
            cur, cur_start = [], w["start"]
        cur.append(w)
    if cur:
        pieces.append(_mk_piece(unit["speaker"], cur))
    return pieces


def window_turns(units: list[dict], max_dur: float,
                 max_turns: int | None = None) -> list[list[dict]]:
    """Greedily group turns into windows bounded by max_dur seconds and, optionally,
    max_turns speaker turns.

    Turns are never cut across a window boundary; a single turn longer than
    max_dur is first split at word boundaries by _split_long_turn."""
    windows: list[list[dict]] = []
    cur: list[dict] = []
    cur_start = None
    for u in units:
        for piece in _split_long_turn(u, max_dur):
            fits = (cur
                    and piece["end"] - cur_start <= max_dur
                    and (max_turns is None or len(cur) < max_turns))
            if fits:
                cur.append(piece)
            else:
                if cur:
                    windows.append(cur)
                cur, cur_start = [piece], piece["start"]
    if cur:
        windows.append(cur)
    return windows


def _pieces_to_segs(pieces: list[dict], win_start: float):
    """Render a list of turn pieces into (segs, labels), numbering speakers 1, 2, ...
    by order of first appearance among *these* pieces."""
    labels: dict[str, int] = {}
    segs = []
    for p in pieces:
        if p["speaker"] not in labels:
            labels[p["speaker"]] = len(labels) + 1
        segs.append({
            "n": labels[p["speaker"]],
            "s": round(p["start"] - win_start, 2),
            "e": round(p["end"] - win_start, 2),
            "text": p["text"],
        })
    return segs, labels


def build_diar_rows(meeting_id: str, mix_audio: Path, units: list[dict],
                    max_dur: float, min_turns: int, min_dur: float,
                    max_turns: int | None = None, split: str | None = None,
                    format_variety: bool = True, backchannel_versions: bool = True,
                    generic_ratio: float = 0.4, backchannel_ratio: float = 0.5,
                    mixed_variants: tuple | None = None) -> dict:
    """Build diarization rows for every variant from one shared windowing.

    For each (window, variant, version) a (format, prompt_style) is chosen by
    prompts_diarization.choose_format: JSON is emitted for a small fixed fraction of rows
    (JSON_FORMAT_RATIO), otherwise with probability generic_ratio a generic,
    no-format prompt on the variant's DEFAULT format, else an explicit
    format-specifying prompt on a random non-JSON format. The choice is stored in
    custom_metadata so make_diar_lean_row can pick a matching prompt.

    When backchannel_versions, each window yields a "clean" version
    (backchannels/acknowledgements removed, default prompt); if it also contained
    backchannels, a "full" version (kept, prompt asks to include them) is emitted
    with probability backchannel_ratio. Windows entirely of backchannels yield
    only the full version. Windows with fewer than min_turns turns or shorter than
    min_dur seconds are dropped. The audio clip is identical across versions."""
    if split is None:
        split = split_for(meeting_id)
    out: dict[str, list] = {v: [] for v in DIAR_VARIANTS}
    if mixed_variants:
        out["mixed"] = []
    for i, window in enumerate(window_turns(units, max_dur, max_turns)):
        if len(window) < min_turns:
            continue
        win_start = window[0]["start"]
        win_end = max(p["end"] for p in window)
        dur = win_end - win_start
        if dur < min_dur or dur <= 0:
            continue

        # (tag, pieces, include_backchannels). Audio window is unchanged; only text differs.
        if backchannel_versions:
            clean = clean_window_pieces(window)
            versions = []
            if clean:
                versions.append(("clean", clean, False))
            if not clean:
                versions.append(("full", window, True))
            elif len(clean) < len(window):
                rng_bc = random.Random(f"{meeting_id}.{i}.bc")
                if rng_bc.random() < backchannel_ratio:
                    versions.append(("full", window, True))
        else:
            versions = [("all", window, None)]

        for tag, pieces, include_bc in versions:
            segs, labels = _pieces_to_segs(pieces, win_start)
            if not segs:
                continue
            seg_meta = [{
                "speaker_letter": p["speaker"],
                "label": f"Speaker {labels[p['speaker']]}",
                "start": round(p["start"], 3),
                "end": round(p["end"], 3),
                "text": p["text"],
            } for p in pieces]
            built: dict[str, NemoDatasetRow] = {}
            for variant in DIAR_VARIANTS:
                formats = _DIAR_FORMATS[variant]
                rng = random.Random(f"{meeting_id}.{i}.{variant}.{tag}")
                fmt, prompt_style = choose_format(variant, formats, rng,
                                                  format_variety, generic_ratio)
                target = formats[fmt]["render"](segs, meeting_id)
                if not target.strip():
                    continue
                audio_turn = NemoTurn(role="User", value=str(mix_audio), turn_type="audio",
                                      duration=round(dur, 3), offset=round(win_start, 3))
                target_turn = NemoTurn(role="Assistant", value=target, turn_type="text")
                md = {
                    "meeting": meeting_id,
                    "variant": variant,
                    "format": fmt,
                    "prompt_style": prompt_style,
                    "num_speakers": len(labels),
                    "segments": seg_meta,
                }
                if include_bc is not None:
                    md["backchannels"] = include_bc
                row = NemoDatasetRow(
                    id=f"{meeting_id}.diar.{i}.{tag}",
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[audio_turn, target_turn],
                    custom_metadata=md,
                )
                out[variant].append(row)
                built[variant] = row
            # 'mixed': one row per window, rendered as a single randomly-chosen variant.
            if mixed_variants:
                pool = [v for v in mixed_variants if v in built]
                if pool:
                    rng_m = random.Random(f"{meeting_id}.{i}.{tag}.mixed")
                    out["mixed"].append(built[rng_m.choice(pool)])
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert ICSI (NXT layout) to NeMo manifests.")
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--root", type=str, default=None,
                          help="ICSI download root (contains Signals/ and annotations/).")
    io_group.add_argument("--raw-manifest-path", type=str, default=None,
                          help="Override raw manifest output folder (with custom_metadata).")
    io_group.add_argument("--manifest-path", type=str, default=None,
                          help="Override metadata-free manifest output folder.")
    io_group.add_argument("--force", action="store_true",
                          help="Overwrite existing manifest .jsonl files.")
    io_group.add_argument("--limit", type=int, default=None,
                          help="Process at most N meetings (debug).")

    summary_group = parser.add_argument_group("summary manifest")
    summary_group.add_argument("--summary", action="store_true",
                               help="Enable the summarization manifest (off by default).")
    summary_group.add_argument("--summary-segment", choices=["none", "topics", "turns"], default="topics",
                               help="'none' = one row per meeting (full interaction.wav). "
                                    "'topics' = one row per topic, with offset/duration "
                                    "and topic-attributed abstractive sentences. "
                                    "'turns' = one row per abstractive sentence, audio span "
                                    "from linked dialogue acts.")
    summary_group.add_argument("--summary-per-section", action="store_true",
                               help="Emit separate rows per summary section "
                                    "(abstract/decisions/problems/progress). "
                                    "Works with 'none' and 'topics' segment modes.")
    summary_group.add_argument("--summary-min-duration", type=float, default=10.0)
    summary_group.add_argument("--summary-max-duration", type=float, default=900.0)

    asr_group = parser.add_argument_group("asr manifest")
    asr_group.add_argument("--asr", action="store_true",
                           help="Enable the per-segment ASR manifest (off by default).")
    asr_group.add_argument("--asr-min-duration", type=float, default=0.2)
    asr_group.add_argument("--asr-max-duration", type=float, default=60.0)

    diar_group = parser.add_argument_group("diarization manifest")
    diar_group.add_argument("--diarization", action="store_true",
                            help="Enable the speaker-diarization manifests (off by default). "
                                 "Emits three variants under diarization/: 'asr' "
                                 "(speaker-labeled transcript), 'timestamps' (who-spoke-when), "
                                 "and 'timestamps_asr' (labels + times + words).")
    diar_group.add_argument("--diar-max-duration", type=float, default=90.0,
                            help="Max window length in seconds. Speaker turns are packed into "
                                 "windows without being cut, except a single turn longer than "
                                 "this is split at word boundaries.")
    diar_group.add_argument("--diar-min-duration", type=float, default=10.0,
                            help="Drop diarization windows shorter than this many seconds.")
    diar_group.add_argument("--diar-min-turns", type=int, default=3,
                            help="Drop windows with fewer than this many speaker turns "
                                 "(set 1 to keep long single-turn windows).")
    diar_group.add_argument("--diar-max-turns", type=int, default=10,
                            help="Start a new window once it reaches this many speaker turns, "
                                 "even if the duration cap is not hit (0 disables the cap).")
    diar_group.add_argument("--diar-format-variety", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="Vary the output format per row (RTTM, 'N: words', bracketed "
                                 "times, ...) with a prompt that specifies it, to teach "
                                 "format-following. --no-diar-format-variety forces the default "
                                 "format for every row.")
    diar_group.add_argument("--diar-generic-prompt-ratio", type=float, default=0.5,
                            help="Fraction of rows that use a generic, no-format prompt "
                                 "(rendered in the default format) instead of an explicit "
                                 "format-specifying prompt.")
    diar_group.add_argument("--diar-cross-lingual-prompt-ratio", type=float, default=0.05,
                            help="Probability of drawing the prompt from a different language than "
                                 "the audio (e.g. a French prompt on English data and vice versa); "
                                 "only formats defined in both languages are eligible.")
    diar_group.add_argument("--diar-backchannel-versions", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="Emit two versions per window: one without "
                                 "backchannels/acknowledgements (default prompt) and, when the "
                                 "window has any, one with them (prompt asks to include them). "
                                 "--no-diar-backchannel-versions keeps every utterance in a "
                                 "single version.")
    diar_group.add_argument("--diar-backchannel-ratio", type=float, default=0.5,
                            help="When a window contains backchannels, the probability of also "
                                 "emitting its backchannel-included version.")
    diar_group.add_argument("--diar-mixed", action=argparse.BooleanOptionalAction, default=True,
                            help="Also emit a 'mixed/' folder where each window is a single row "
                                 "rendered as one randomly-chosen variant (sampled from "
                                 "--diar-mixed-variants).")
    diar_group.add_argument("--diar-mixed-variants", nargs="+", choices=list(DIAR_VARIANTS),
                            default=list(DIAR_VARIANTS),
                            help="Pool of variants to sample from for the 'mixed' folder "
                                 "(default: all three; pass two for a 2-of-3 mix).")
    diar_group.add_argument("--diar-merge-gap", type=float, default=1.0,
                            help="Merge consecutive same-speaker turns separated by <= this "
                                 "many seconds. Negative disables merging.")
    diar_group.add_argument("--diar-raw-manifest-path", type=str, default=None,
                            help="Raw diarization manifest folder (with custom_metadata). "
                                 "Kept next to the audio by default: "
                                 "$DATA_FOLDER/raw/summary/en/icsi/diarization.")
    diar_group.add_argument("--diar-manifest-path", type=str, default=None,
                            help="Training diarization manifest folder. "
                                 "Default: $DATA_FOLDER/nemo/diarization/en/icsi.")

    args = parser.parse_args()

    if not (args.summary or args.asr or args.diarization):
        parser.error("Nothing to do: pass at least one of --summary, --asr, --diarization.")

    if args.root is None:
        args.root = f"{os.environ['DATA_FOLDER']}/raw/summary/en/icsi"
        print(f"Input path not specified, using default: {args.root}")
    root = Path(args.root)
    signals_dir = root / "Signals"
    ann_dir = root / "annotations" / "ICSIplus"
    abs_dir = ann_dir / "Contributions" / "Summarization" / "abstractive"
    extractive_dir = ann_dir / "Contributions" / "Summarization" / "extractive"
    topics_dir = ann_dir / "Contributions" / "TopicSegmentation"
    words_dir = ann_dir / "Words"
    seg_dir = ann_dir / "Segments"
    dialog_acts_dir = ann_dir / "DialogueActs"

    for d in (signals_dir, ann_dir):
        if not d.is_dir():
            parser.error(f"Missing directory: {d}")

    data_dir = os.environ.get("DATA_FOLDER")
    if args.raw_manifest_path is None:
        if not data_dir:
            parser.error("--raw-manifest-path not set and DATA_FOLDER env var is missing")
        args.raw_manifest_path = f"{data_dir}/raw/summary/en/icsi"
    if args.manifest_path is None:
        if not data_dir:
            parser.error("--manifest-path not set and DATA_FOLDER env var is missing")
        args.manifest_path = f"{data_dir}/nemo/summary/en/icsi"

    if args.diarization:
        if args.diar_raw_manifest_path is None:
            if not data_dir:
                parser.error("--diar-raw-manifest-path not set and DATA_FOLDER env var is missing")
            args.diar_raw_manifest_path = f"{data_dir}/raw/summary/en/icsi/diarization"
        if args.diar_manifest_path is None:
            if not data_dir:
                parser.error("--diar-manifest-path not set and DATA_FOLDER env var is missing")
            args.diar_manifest_path = f"{data_dir}/nemo/diarization/en/icsi"

    raw_root = Path(args.raw_manifest_path)
    out_root = Path(args.manifest_path)
    diar_raw_root = Path(args.diar_raw_manifest_path) if args.diarization else None
    diar_out_root = Path(args.diar_manifest_path) if args.diarization else None
    if args.summary:
        (raw_root / "summary").mkdir(parents=True, exist_ok=True)
        (out_root / "summary").mkdir(parents=True, exist_ok=True)
    if args.asr:
        (raw_root / "asr").mkdir(parents=True, exist_ok=True)
        (out_root / "asr").mkdir(parents=True, exist_ok=True)
    diar_outputs = list(DIAR_VARIANTS) + (["mixed"] if (args.diarization and args.diar_mixed) else [])
    if args.diarization:
        for v in diar_outputs:
            (diar_raw_root / v).mkdir(parents=True, exist_ok=True)
            (diar_out_root / v).mkdir(parents=True, exist_ok=True)

    meetings = discover_meetings(signals_dir)
    if args.limit:
        meetings = meetings[:args.limit]
    logger.info(f"Found {len(meetings)} meetings under {signals_dir}")

    summary_per_split: dict[str, tuple[NemoDataset, NemoDataset]] = {}
    asr_per_split: dict[str, tuple[NemoDataset, NemoDataset]] = {}
    diar_buckets: dict[str, dict[str, tuple[NemoDataset, NemoDataset]]] = {
        v: {} for v in diar_outputs
    }

    def get(buckets, split):
        if split not in buckets:
            buckets[split] = (NemoDataset(name=DATASET_NAME), NemoDataset(name=DATASET_NAME))
        return buckets[split]

    def make_lean_row(r: NemoDatasetRow) -> NemoDatasetRow:
        turns = list(r.turns)
        section = (r.custom_metadata or {}).get("section")
        if section and section in _SECTION_PROMPTS:
            text = random.choice(_SECTION_PROMPTS[section])
            prompt = NemoTurn(role="User", value=text, turn_type="text")
            turns = [prompt] + turns
        return NemoDatasetRow(
            id=r.id, dataset_name=r.dataset_name,
            split=r.split, language=r.language, turns=turns,
        )

    for meeting_id in tqdm(meetings, desc="meetings"):
        mix = signals_dir / meeting_id / f"{meeting_id}.interaction.wav"
        if not mix.exists():
            logger.warning(f"{meeting_id}: no interaction.wav, skipping")
            continue
        split = split_for(meeting_id)

        mix_dur = audio_duration(mix)
        if mix_dur is None:
            continue

        if args.summary:
            if args.summary_segment == "topics":
                topic_rows = build_topic_summary_rows(
                    meeting_id, mix, words_dir, seg_dir,
                    dialog_acts_dir, abs_dir, extractive_dir, topics_dir,
                    mix_dur, args.summary_min_duration, args.summary_max_duration,
                    per_section=args.summary_per_section,
                )
                if topic_rows:
                    raw, lean = get(summary_per_split, split)
                    for r in topic_rows:
                        raw.append(r)
                        lean.append(make_lean_row(r))
            elif args.summary_segment == "turns":
                turn_rows = build_turn_summary_rows(
                    meeting_id, mix, words_dir, dialog_acts_dir, abs_dir,
                    extractive_dir,
                    mix_dur, args.summary_min_duration, args.summary_max_duration,
                )
                if turn_rows:
                    raw, lean = get(summary_per_split, split)
                    for r in turn_rows:
                        raw.append(r)
                        lean.append(make_lean_row(r))
            else:
                abs_xml = abs_dir / f"{meeting_id}.abssumm.xml"
                if not abs_xml.exists():
                    logger.debug(f"{meeting_id}: no abstractive summary")
                else:
                    try:
                        summary = parse_abstractive_summary(abs_xml)
                    except ET.ParseError as e:
                        logger.warning(f"{meeting_id}: bad abssumm.xml ({e})")
                        summary = None
                    if summary is not None:
                        if args.summary_per_section:
                            section_rows = build_section_summary_rows(
                                meeting_id, mix, summary, mix_dur,
                            )
                            if section_rows:
                                raw, lean = get(summary_per_split, split)
                                for r in section_rows:
                                    raw.append(r)
                                    lean.append(make_lean_row(r))
                        else:
                            row = build_summary_row(meeting_id, mix, summary, mix_dur)
                            if row is not None:
                                raw, lean = get(summary_per_split, split)
                                raw.append(row)
                                lean.append(make_lean_row(row))

        if args.asr:
            rows = build_asr_rows(
                meeting_id, mix, words_dir, seg_dir, mix_dur,
                args.asr_min_duration, args.asr_max_duration,
            )
            if rows:
                raw, lean = get(asr_per_split, split)
                for r in rows:
                    raw.append(r)
                    lean.append(NemoDatasetRow(
                        id=r.id, dataset_name=r.dataset_name, split=r.split,
                        language=r.language, speaker=r.speaker, turns=r.turns,
                    ))

        if args.diarization:
            units = collect_meeting_turns(
                meeting_id, words_dir, seg_dir,
                None if args.diar_merge_gap < 0 else args.diar_merge_gap,
                mix_duration=mix_dur,
            )
            if units:
                per_variant = build_diar_rows(
                    meeting_id, mix, units,
                    args.diar_max_duration, args.diar_min_turns, args.diar_min_duration,
                    max_turns=None if args.diar_max_turns <= 0 else args.diar_max_turns,
                    split=split, format_variety=args.diar_format_variety,
                    backchannel_versions=args.diar_backchannel_versions,
                    generic_ratio=args.diar_generic_prompt_ratio,
                    backchannel_ratio=args.diar_backchannel_ratio,
                    mixed_variants=tuple(args.diar_mixed_variants) if args.diar_mixed else None,
                )
                for variant, rows in per_variant.items():
                    if not rows:
                        continue
                    raw, lean = get(diar_buckets[variant], split)
                    for r in rows:
                        raw.append(r)
                        lean.append(make_diar_lean_row(r, args.diar_cross_lingual_prompt_ratio))

    def dump(buckets, kind, raw_base=raw_root, out_base=out_root):
        for split, (raw, lean) in buckets.items():
            if not len(raw):
                continue
            raw_file = raw_base / kind / f"{split}.jsonl"
            lean_file = out_base / kind / f"{split}.jsonl"
            if raw_file.exists() and lean_file.exists() and not args.force:
                logger.info(f"[{kind}/{split}] exists, skipping (use --force)")
                continue
            raw.save(raw_file)
            lean.save(lean_file)
            logger.info(f"[{kind}/{split}] wrote {len(raw)} rows → {raw_file}")
            logger.info(f"[{kind}/{split}] wrote {len(lean)} rows → {lean_file}")

    if args.summary:
        dump(summary_per_split, "summary")
    if args.asr:
        dump(asr_per_split, "asr")
    if args.diarization:
        for variant in diar_outputs:
            dump(diar_buckets[variant], variant,
                 raw_base=diar_raw_root, out_base=diar_out_root)


if __name__ == "__main__":
    main()
