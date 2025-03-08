"""The transcription_result module holds classes responsible for holding, merging and formating transcription results."""
# This is a copy of https://github.com/linto-ai/linto-platform-transcription-service/blob/master/transcriptionservice/transcription/transcription_result.py
import json
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class Word:
    """Contains word informations"""

    word: str
    start: float
    end: float
    conf: float

    def apply_offset(self, offset: float):
        self.start += offset
        self.end += offset

    @property
    def json(self) -> dict:
        return self.__dict__


@dataclass
class DiarizationSegment:
    """Contain diarization segment information"""

    seg_begin: float
    seg_end: float
    spk_id: Any
    seg_id: int

    @property
    def json(self) -> dict:
        return self.__dict__


@dataclass
class SpeechSegment:
    speaker_id: str = None
    words: list = field(default_factory=list)
    processed_segment = None

    def toString(self, include_spkid: bool = False, spk_sep: str = ":"):
        output = f"{self.speaker_id}{spk_sep} " if include_spkid and self.speaker_id is not None else ""
        return output + (self.raw_segment if self.processed_segment is None else self.processed_segment)

    @property
    def raw_segment(self) -> str:
        return " ".join([w.word for w in self.words]).strip()

    @property
    def start(self) -> float:
        return min([w.start for w in self.words]) if len(self.words) > 0 else 0.0

    @property
    def end(self) -> float:
        return max([w.end for w in self.words]) if len(self.words) > 0 else 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def json(self) -> dict:
        return {
            "spk_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "raw_segment": self.raw_segment,
            "segment": self.processed_segment if self.processed_segment is not None else self.raw_segment,
            "words": [w.json for w in self.words],
        }


class TranscriptionResult:
    """Transcription result manages transcription results, post-processing and formating for transcription results."""

    def __init__(self, transcriptions: list[tuple[dict, float]], spk_ids: list = None):
        """Initialisation accepts list of tuple (transcription, time_offset)"""
        self.transcription_confidence = 0.0
        self.words = []
        self.segments = []
        self.diarizationSegments = []
        if transcriptions:
            self._mergeTranscription(transcriptions, spk_ids)

    def _mergeTranscription(self, transcriptions: list[tuple[dict, float]], spk_ids: list = None) -> None:
        """Merges transcription results applying offsets"""
        self.transcription_confidence = 0.0
        num_words = 0
        for transcription, offset in transcriptions:
            for w in transcription["words"]:
                word = Word(**w)
                word.apply_offset(offset)
                self.words.append(word)
                self.transcription_confidence += word.conf
            num_words += len(transcription["words"])
        if num_words:
            self.transcription_confidence /= num_words
        self.words.sort(key=lambda x: x.start)

        if spk_ids:
            for (transcription, offset), id in zip(transcriptions, spk_ids):
                seg_words = []
                for w in transcription["words"]:
                    word = Word(**w)
                    word.apply_offset(offset)
                    seg_words.append(word)
                if seg_words:
                    self.segments.append(SpeechSegment(id, seg_words))
            self.segments = sorted(self.segments, key=lambda seg: seg.start)

    def setTranscription(self, words: list[dict]):
        for w in words:
            self.words.append(Word(**w))
        self.transcription_confidence = sum([w.conf for w in self.words]) / len(self.words) if len(self.words) else 0.0

    def setDiarizationResult(self, diarizationResult: Union[str, dict]):
        """Create speech segments using word and diarization data"""
        diarization_data = json.loads(diarizationResult) if isinstance(diarizationResult, str) else diarizationResult

        # Get segments ordered by start time
        self.diarizationSegments = [DiarizationSegment(**segment) for segment in sorted(diarization_data["segments"], key=lambda x: x["seg_begin"])]

        # Filter out segments that are included in others (i.e. which end before previous segment)
        self.diarizationSegments = [self.diarizationSegments[i] for i in range(len(self.diarizationSegments)) if i == 0 or self.diarizationSegments[i].seg_end > self.diarizationSegments[i - 1].seg_end]

        self.words.sort(key=lambda x: x.start)

        # Interpolates speaker change timestamps
        # Starts the first segment at 0.0, ends the last segment at max(word.ends)
        # If two consecutive diarization segments are not joint, find the middle point
        self.diarizationSegments[0].seg_begin = 0.0
        if len(self.words):
            self.diarizationSegments[-1].seg_end = max(self.diarizationSegments[-1].seg_end, self.words[-1].end)
        for first_segment, second_segment in zip(self.diarizationSegments[:-1], self.diarizationSegments[1:]):
            # When there is a gap or an overlap between the two segments
            if first_segment.seg_end != second_segment.seg_begin:
                middle_point = first_segment.seg_end + (second_segment.seg_begin - first_segment.seg_end) / 2
                first_segment.seg_end = second_segment.seg_begin = middle_point

        seg_index = 0
        previous_id = None
        current_id = self.diarizationSegments[seg_index].spk_id
        current_words = []

        # Iterate over segments and words to create speech segments
        for i, word in enumerate(self.words):
            while not self._resolveWordSegment(i, seg_index):
                # Next segment
                if seg_index + 1 < len(self.diarizationSegments):
                    seg_index += 1
                    next_id = self.diarizationSegments[seg_index].spk_id
                else:
                    break
                if len(current_words):
                    if current_id != previous_id:  # Flush current segment
                        self.segments.append(SpeechSegment(current_id, current_words))
                    else:  # Merge with previous segment
                        self.segments[-1].words += current_words
                    previous_id = current_id
                current_id = next_id
                current_words = []
            current_words.append(word)
        if len(current_words):
            if current_id != previous_id:  # Flush current segment
                self.segments.append(SpeechSegment(current_id, current_words))
            else:  # Merge with previous segment
                self.segments[-1].words += current_words

    def _resolveWordSegment(
        self,
        word_index: int,
        diarization_index: int,
        precision: float = 0.25,
    ) -> bool:
        """
        Applies word placement rules and decides if the word belong to the current_segment (True) or to the next (False)

        Args:
            word_index (int): Index of the word to place
            diarization_index (int): Index of the current diarization segment
            precision (float): Precision to decide if a word is within a segment (in seconds)
        """
        if diarization_index == len(self.diarizationSegments) - 1:
            # Stay on current segment if it is the last one
            return True

        word = self.words[word_index]
        word_start = word.start
        word_end = word.end
        current_diarization_seg = self.diarizationSegments[diarization_index]

        # Word completely within current segment
        if word_end <= current_diarization_seg.seg_end - precision:
            return True

        # Word completely outside current segment
        if word_start >= current_diarization_seg.seg_end + precision:
            return False

        # Otherwise (following), word straddling two segments

        if not word_index:
            # Assign first word to first segment
            return True
        if word_index == len(self.words) - 1:
            # Assign last word to last segment
            return False

        # Decide based on the distance with the previous and the next words
        # if one exceeds a certain threshold seconds
        gap_previous_word = word_start - self.words[word_index - 1].end
        gap_next_word = self.words[word_index + 1].start - word_end
        if max(gap_previous_word, gap_next_word) >= precision:
            return gap_previous_word <= gap_next_word

        if word_index > 0:
            # If the previous word ends with a punctuation, cut there
            previous_word = self.words[word_index - 1]
            if previous_word.word and previous_word.word[-1] in ".!?":
                return False
            elif word.word and word.word[-1] in ".!?":
                return True

        # Otherwise, look at what happens with the next segment
        next_diarization_seg = self.diarizationSegments[diarization_index + 1]
        word_len = word_end - word_start
        overlap_previous = current_diarization_seg.seg_end - word_start
        overlap_next = word_end - next_diarization_seg.seg_begin

        # Should we do something more when a segment is really to short?
        # new_segment_len = next_diarization_seg.seg_end - next_diarization_seg.seg_begin
        # current_segment_len = current_diarization_seg.seg_end - current_diarization_seg.seg_begin
        # current_segment_is_short = current_segment_len < word_len * 2
        # next_segment_is_short = new_segment_len < word_len * 2
        # if current_segment_is_short and not next_segment_is_short:
        #     # Tend to assign words to the next segment if it is short
        #     return True
        # if next_segment_is_short and not current_segment_is_short:
        #     # Tend to assign words to the current segment if it is short
        #     return False

        # Assign to the segment with the higher overlap
        return overlap_previous > overlap_next

    def setNoDiarization(self):
        """Convert word data into a speech segment when there is no diarization"""
        self.segments.append(SpeechSegment(None, self.words))

    def setProcessedSegment(self, processed_segments: Union[list[str], str]):
        """Add the processed_segment value to segments"""
        if type(processed_segments) == str:
            processed_segments = [processed_segments]
        for seg, proc_seg in zip(self.segments, processed_segments):
            seg.processed_segment = proc_seg

    @property
    def final_transcription(self) -> str:
        return " \n".join([seg.toString(include_spkid=True) for seg in self.segments]).strip()

    @property
    def raw_transcription(self) -> str:
        """Return the raw transcription.

        Returns:
            str: Transcription without any other processing
        """
        return " ".join([w.word for w in self.words]).strip()

    @classmethod
    def fromDict(cls, resultDict: dict):
        """Create TranscriptionResult from dictionnary"""
        result = TranscriptionResult(None)
        result.transcription_confidence = resultDict["confidence"]
        for segment in resultDict["segments"]:
            seg = SpeechSegment(segment["spk_id"], [Word(**w) for w in segment["words"]])
            seg.processed_segment = segment["segment"]
            result.segments.append(seg)

        result.diarizationSegments = [DiarizationSegment(**diarizationSegment) for diarizationSegment in resultDict["diarization_segments"]]

        return result

    def final_result(self) -> dict:
        return {
            "transcription_result": self.final_transcription,
            "raw_transcription": self.raw_transcription,
            "confidence": self.transcription_confidence,
            "segments": [s.json for s in self.segments],
            "diarization_segments": [seg.json for seg in self.diarizationSegments],
        }
