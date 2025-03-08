import argparse
import logging
import os
import re
import shutil

from bs4 import BeautifulSoup

from ssak.utils.kaldi_converter import AudioFolder2Kaldi, Reader2Kaldi, Row2Info, ToKaldi
from tools.clean_text_fr import clean_text_fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Xml2Kaldi(ToKaldi):
    def __init__(
        self,
        input,
        return_columns,
        execute_order,
        merge_on="id",
        sort_merging=True,
        max_speakers=2,
        extension=".xml",
        subfolders=True,
    ) -> None:
        if return_columns is None:
            raise ValueError("Columns must be specified")
        super().__init__(input, return_columns, execute_order, merge_on, sort_merging=sort_merging)
        self.subfolders = subfolders
        self.speakers = {}
        self.max_speakers = max_speakers
        self.extension = extension

    def flush_rows(self, data, tmp_rows, timecodes, turn):
        if len(tmp_rows) > 0:
            if len(tmp_rows) <= self.max_speakers:
                end = timecodes[turn.get("synch").strip("# ")]
                for i in tmp_rows:
                    i["end"] = end
                    regex = re.compile(r"[\s|hm|chri]+")
                    t = regex.sub("", i["text"])  # remove 'hm' and 'chri' from the text because lot of segments with just that
                    if len(t) > 0:  # if the text is not empty (after removing 'hm' and 'chri')
                        data.append(i)
            tmp_rows = []
        start = timecodes[turn.get("synch").strip("#")]
        return data, tmp_rows, start

    def process_turn(self, data, tmp_rows, turn, timecodes, file_id, speaker, ct, start):
        text = ""
        for elem in turn.children:
            if elem.name == "anchor":
                if elem.get("synch") is not None and elem.get("synch").startswith("#T") and len(text) > 1:
                    tmp_rows.append(
                        {
                            "id": f"{file_id}_{speaker}_{ct}",
                            "text": text.strip().encode("utf-8").decode("utf-8"),
                            "audio_id": file_id,
                            "start": start,
                            "end": None,
                            "speaker": speaker,
                        }
                    )
                    text = ""
                    ct += 1
                if elem.get("synch") is not None and elem.get("synch").startswith("#T") and len(tmp_rows) > 0:
                    try:
                        data, tmp_rows, start = self.flush_rows(data, tmp_rows, timecodes, elem)
                    except Exception as e:
                        raise ValueError(f"Error in file {file_id}") from e
            elif elem.name == "w":
                text += " " + elem.text
            elif elem.name == "choice":
                text += " " + elem.find("reg").text
            elif elem.name == "seg":
                data, tmp_rows, ct, start = self.process_turn(data, tmp_rows, elem, timecodes, file_id, speaker, ct, start)
        if len(text) > 1:
            tmp_rows.append(
                {
                    "id": f"{file_id}_{speaker}_{ct}",
                    "text": text.strip().encode("utf-8").decode("utf-8"),
                    "audio_id": file_id,
                    "start": start,
                    "end": None,
                    "speaker": speaker,
                }
            )
            text = ""
            ct += 1
        return data, tmp_rows, ct, start

    def parse_file(self, data, soup, file_id=None):
        for person in soup.find_all("person"):
            speaker = person.get("xml:id")
            if speaker not in self.speakers:
                self.speakers[speaker] = "f" if person.get("sex") == "2" else "m"
        timecodes = {}
        timeline = soup.find("timeline")
        tenth = False
        bad_timeline_annotaions = False
        if timeline is None:
            return data
        for i in timeline.find_all("when"):
            if not i.text.isspace():
                hours, minutes, seconds = i.get("absolute").split(":")
                if i.get("xml:id") == "T1" and seconds == "00.00" and minutes == "00" and hours == "00":
                    bad_timeline_annotaions = True
                    continue
                split_seconds = seconds.split(".")
                if len(split_seconds) > 1:
                    if len(split_seconds[1]) == 1:
                        tenth = True
                    if tenth:
                        seconds = float(split_seconds[0]) + 1 if split_seconds[1] == "10" else seconds
                id = i.get("xml:id")
                if bad_timeline_annotaions:
                    id = f"T{int(id[1:])-1}"
                timecodes[id] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        body = soup.find("body")
        start = 0
        speaker = None
        ct = 0
        tmp_rows = []
        for turn in body.children:
            if not turn.text.isspace():
                if turn.name == "anchor":
                    try:
                        data, tmp_rows, start = self.flush_rows(data, tmp_rows, timecodes, turn)
                    except Exception as e:
                        raise ValueError(f"Error in file {file_id}") from e
                elif turn.name == "u":
                    speaker = turn.get("who").strip("#")
                    data, tmp_rows, ct, start = self.process_turn(data, tmp_rows, turn, timecodes, file_id, speaker, ct, start)
        return data

    def process(self, dataset):
        files_to_look_for = None
        if len(dataset) > 0:
            files_to_look_for = set([f"{i['audio_id']}" for i in dataset])
        data = []
        for root, _, files in os.walk(self.input):
            for file in files:
                file_id = "_".join(os.path.splitext(file)[0].split("_")[0:-1])
                if file.endswith(self.extension) and (files_to_look_for is None or file_id in files_to_look_for):
                    with open(os.path.join(root, file), encoding="ISO-8859-1") as tei:
                        soup = BeautifulSoup(tei, features="xml", from_encoding="ISO-8859-1")
                        data = self.parse_file(data, soup, file_id=file_id)
            if not self.subfolders:
                break
        return self.merge_data(dataset, new_data=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CLAPI dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/CLAPI/clapi/2/CorpusComplet")
    parser.add_argument("--output", type=str, default="/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/CLAPI")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = args.output

    raw = os.path.join(output_path, "raw")

    nocasepunc = os.path.join(output_path, "nocasepunc")

    if os.path.exists(nocasepunc) and not args.force:
        raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    elif os.path.exists(nocasepunc):
        shutil.rmtree(nocasepunc)

    audios = AudioFolder2Kaldi(input_dataset, execute_order=0, extracted_id="audio_id", audio_extensions=[".mp3"])
    audios_ids = Row2Info("audio_id", ["audio_id", "id"], execute_order=1, separator="_", info_position=[0, -2])
    xmls = Xml2Kaldi(input_dataset, ["id", "text", "start", "duration"], execute_order=2, merge_on="audio_id", subfolders=True)
    dataset_reader = Reader2Kaldi(input_dataset, processors=[audios, audios_ids, xmls])
    dataset = dataset_reader.load(check_if_segments_in_audio=True)
    dataset.save(raw, True)

    clean_text_fr(raw, nocasepunc, ignore_first=1, file_clean_mode="kaldi")
