# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import logging
from lhotse import CutSet
from random import Random
from tqdm import tqdm
import soundfile as sf
import numpy as np
import logging
import torch
import glob
from collections import deque
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence, Union
from lhotse import CutSet, AudioSource, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioLoadingError
from lhotse.custom import CustomFieldMixin
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_matrices, collate_vectors
from lhotse.dataset.dataloading import resolve_seed
from lhotse.serialization import load_jsonl
from lhotse.shar import AudioTarWriter, JsonlShardWriter
from lhotse.utils import Pathlike, is_valid_url
from lhotse.utils import compute_num_samples, ifnone

from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationTarWriter,
    TextTurn,
    AudioTurn,
    get_full_path,
    NeMoMultimodalConversation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomAudioTurn(AudioTurn):

    def to_dict(self):
        assert self.cut.has_recording and self.cut.recording.sources[0].type not in {
            "shar",
            "memory",
        }, "Cannot serialize AudioTurn to dict because it doesn't reference an audio file (the audio is stored in memory)."
        return {
            "type": "audio",
            "from": self.role.title(),
            "duration": self.cut.duration,
            # "id": self.cut.id,
            "id": self.cut.recording.id,
            "value": self.cut.recording.sources[0].source,
            "text": self.text,
        }

@dataclass
class CustomNeMoMultimodalConversationJsonlAdapter(NeMoMultimodalConversationJsonlAdapter):

    def _iter_jsonl(self):
        paths = self.manifest_filepath
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                if self._should_skip(data):
                    continue
                turns = []
                for turn in data["conversations"]:
                    if turn["type"] == "text":
                        turns.append(TextTurn(value=turn["value"], role=turn["from"].lower()))
                    else: 
                        # recording = Recording.from_file(get_full_path(turn["value"], path))
                        recording_id = Path(get_full_path(turn["value"], path)).stem
                        duration = round(turn["duration"], 3)
                        offset = round(turn["offset"], 3)
                        cut_id = f"{recording_id}_{offset:.2f}".replace(".", "_")
                        cut = MonoCut(
                            id=cut_id,
                            start=offset,
                            duration=duration,
                            channel=0,
                            supervisions=[],
                            recording=Recording(
                                id=cut_id,
                                sources=[AudioSource(type="file", channels=[0], source=get_full_path(turn["value"], path))],
                                sampling_rate=16_000,
                                duration=duration,
                                num_samples=compute_num_samples(duration, 16_000),
                            )
                        )
                        audio_turn = CustomAudioTurn(
                            cut=cut,
                            text=cut.supervisions[0].text if cut.supervisions else None,
                            role=turn["from"].lower(),
                            audio_locator_tag=self.audio_locator_tag,
                        )
                        turns.append(audio_turn)
                if self.system_prompt is not None and turns[0].role != "system":
                    turns = [TextTurn(role="system", value=self.system_prompt)] + turns
                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=turns,
                    token_equivalent_duration=self.token_equivalent_duration,
                    custom=data.get("custom"),
                )

def load_cut_audio_with_offset(cut):
    """
    Load audio from a cut while respecting cut.start and cut.duration.
    Returns numpy array of shape (channels, samples)
    """
    sr = cut.sampling_rate

    start_sample = int(cut.start * sr)
    num_samples = int(cut.duration * sr)

    # Use soundfile to load only the required frames
    audio, file_sr = sf.read(cut.recording.sources[0].source, start=start_sample, frames=num_samples, dtype='float32')

    # Ensure sample rate consistency
    assert file_sr == sr, (
        f"Recording SR ({file_sr}) does not match cut SR ({sr})"
    )

    # SoundFile returns (samples, channels) → transpose to (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]      # (samples,) → (1, samples)
    else:
        audio = audio.T                   # (samples, channels) → (channels, samples)

    return audio


class CustomNeMoMultimodalConversationTarWriter(NeMoMultimodalConversationTarWriter):
    
    def write(self, example):
        self._maybe_increment_shard()
        serialized = example.to_dict()
        for turn in serialized["conversations"]:
            if turn["type"] == "audio":
                turn["value"] = Path(turn["id"]).with_suffix(".flac").name
                turn.pop("id")
        self.manifest_writer.write(serialized)
        for cut in example.list_cuts():
            assert (
                cut.has_recording
            ), f"Cannot serialize multimodal conversation with cuts that have no recordings. We got: {cut}"
            audio =  load_cut_audio_with_offset(cut)
            self.tar_writer.write(cut.id, audio, cut.sampling_rate, cut.recording)
        self.item_cntr += 1


@click.command()
@click.argument("manifest", type=click.Path())
@click.argument("output_dir", type=click.Path())
@click.option("-n", "--shard_size", type=int, default=10_000, help="Number of conversations per shard.")
@click.option("--shuffle/--no-shuffle", default=False, help="Shuffle conversations.")
@click.option("-s", "--seed", type=int, default=42, help="Random seed.")
def export(manifest: str, output_dir: str, shard_size: int, shuffle: bool, seed: int):
    logger.info(f"Exporting conversations from {manifest} to {output_dir}")
    output_dir = Path(output_dir)
    merged_path = output_dir / Path(manifest).name
    if merged_path.exists():
        logger.info(f"Found existing merged manifest at {merged_path}, skipping export.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, 'r') as f:
        number_lines = sum(1 for _ in f)
    with CustomNeMoMultimodalConversationTarWriter(output_dir, shard_size=shard_size) as writer:
        source = CustomNeMoMultimodalConversationJsonlAdapter(manifest, audio_locator_tag="<|audio|>")
        if shuffle:
            source = CutSet(source).shuffle(buffer_size=50000, rng=Random(seed))
        for item in tqdm(source, desc="Writing in .tar files", total=number_lines):
            writer.write(item)
    logger.info(f"Finished exporting to {output_dir}")
    merge_manifest = True
    if merge_manifest:
            logger.info("Merging JSONL shards into a final manifest...")
            jsonl_files = sorted(output_dir.glob("*.jsonl"))
            with open(merged_path, "w") as outfile:
                for file in jsonl_files:
                    with open(file, "r") as infile:
                        for line in infile:
                            outfile.write(line)
            logger.info(f"Merged manifest created at: {merged_path}")

if __name__ == '__main__':
    export()
