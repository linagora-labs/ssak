import os
import time

import numpy as np
import torch
import transformers

from ssak.utils.dataset import kaldi_folder_to_dataset, process_dataset, to_audio_batches
from ssak.utils.misc import flatten

from .utils import Test


class TestAudioDataset(Test):
    def test_kaldi_to_huggingface_dataset(self):
        kaldir = self.get_data_path("kaldi/complete")

        tic = time.time()
        meta, dataset = kaldi_folder_to_dataset(kaldir, verbose=False)
        t = time.time() - tic
        EXPECTED = "4270bec2a6c177683597dbb49c70473b"

        self.check_dataset(dataset)
        self.assertTrue(hasattr(dataset, "__len__"))
        self.assertEqual(len(dataset), 63)
        self.assertEqual(self.hash(list(dataset)), EXPECTED)
        self.assertEqual(meta["samples"], 63)
        self.assertTrue(meta["h duration"] > 0.056 and meta["h duration"] < 0.057, f"Duration is {meta['h duration']}")

        tic = time.time()
        meta_online, dataset_online = kaldi_folder_to_dataset(kaldir, online=True, verbose=False)
        t_online = time.time() - tic

        self.check_dataset(dataset_online)
        self.assertFalse(hasattr(dataset_online, "__len__"))
        self.assertEqual(self.hash(list(dataset_online)), self.hash(list(dataset)))
        self.assertEqual(meta_online["samples"], 63)
        self.assertTrue(meta_online["h duration"] > 0.056 and meta["h duration"] < 0.057, f"Duration is {meta['h duration']}")
        # self.assertGreater(t, t_online) # Not necessarily true

        processor = transformers.Wav2Vec2Processor.from_pretrained("Ilyes/wav2vec2-large-xlsr-53-french")

        tic = time.time()
        processed = process_dataset(processor, dataset, verbose=False)
        t = time.time() - tic
        EXPECTED = "28519edeb9120b98b4baaa4bf15984ee"

        self.check_audio_dataset(processed)
        self.assertTrue(hasattr(dataset, "__len__"))
        self.assertEqual(len(processed), 63)
        self.assertEqual(self.loosehash(list(processed)), EXPECTED)

        tic = time.time()
        processed_online = process_dataset(processor, dataset_online, verbose=False)
        t_online = time.time() - tic

        self.check_audio_dataset(processed_online)
        self.assertFalse(hasattr(processed_online, "__len__"))
        self.assertEqual(self.loosehash(list(processed_online)), EXPECTED)
        # self.assertGreater(t, t_online) # Not necessarily true

    def test_kaldi_to_huggingface_dataset_min_files(self):
        kaldir = self.get_data_path("kaldi/minimal")

        tic = time.time()
        meta, dataset = kaldi_folder_to_dataset(kaldir, verbose=False)
        t = time.time() - tic
        EXPECTED = "31e3bf735009b375abb2b7dbb944d5cc"

        self.check_dataset(dataset, False)
        self.assertTrue(hasattr(dataset, "__len__"))
        self.assertEqual(len(dataset), 4)
        self.assertEqual(self.hash(list(dataset)), EXPECTED)
        self.assertEqual(meta["samples"], 4)
        self.assertTrue(meta["h duration"] > 0.009 and meta["h duration"] < 0.0091, f"Duration is {meta['h duration']}")

        tic = time.time()
        meta_online, dataset_online = kaldi_folder_to_dataset(kaldir, online=True, verbose=False)
        t_online = time.time() - tic

        self.check_dataset(dataset_online, False)
        self.assertFalse(hasattr(dataset_online, "__len__"))
        self.assertEqual(self.hash(list(dataset_online)), self.hash(list(dataset)))
        self.assertEqual(meta_online["samples"], 4)
        self.assertTrue(meta_online["h duration"] > 0.009 and meta["h duration"] < 0.0091, f"Duration is {meta['h duration']}")
        # self.assertGreater(t, t_online) # Not necessarily true

        processor = transformers.Wav2Vec2Processor.from_pretrained("Ilyes/wav2vec2-large-xlsr-53-french")

        tic = time.time()
        processed = process_dataset(processor, dataset, verbose=False)
        t = time.time() - tic
        EXPECTED = "d438ed93ecef7975da646ab94a8c38ba"

        self.check_audio_dataset(processed)
        self.assertTrue(hasattr(dataset, "__len__"))
        self.assertEqual(len(processed), 4)
        self.assertEqual(self.loosehash(list(processed)), EXPECTED)

        tic = time.time()
        processed_online = process_dataset(processor, dataset_online, verbose=False)
        t_online = time.time() - tic

        self.check_audio_dataset(processed_online)
        self.assertFalse(hasattr(processed_online, "__len__"))
        self.assertEqual(self.loosehash(list(processed_online)), EXPECTED)
        # self.assertGreater(t, t_online) # Not necessarily true

    def check_dataset(self, dataset, has_segment=True):
        ids = []
        for data in dataset:
            self.assertTrue(isinstance(data.get("ID"), str))
            if has_segment:
                self.assertTrue(isinstance(data.get("start"), float))
                self.assertTrue(isinstance(data.get("end"), float))
            else:
                # self.assertTrue("start" not in data)
                # self.assertTrue("end" not in data)
                self.assertFalse(data["start"])
                self.assertFalse(data["end"])
            self.assertTrue(isinstance(data.get("text"), str))
            self.assertTrue(isinstance(data.get("path"), str))
            self.assertTrue(os.path.isfile(data.get("path", "")), f"Cannot find {data.get('path')}")
            ids.append(data.get("ID"))
        self.assertEqual(len(ids), len(set(ids)))

    def check_audio_dataset(self, dataset):
        for data in dataset:
            self.assertTrue(
                isinstance(data.get("input_values"), (np.ndarray, list)),
                f"input_values is not an array but: {type(data.get('input_values'))}",
            )
            self.assertGreater(len(data.get("input_values")), 100)
            self.assertTrue(isinstance(data.get("labels"), list))
            self.assertGreater(len(data.get("labels")), 0)
            self.assertTrue(isinstance(data.get("labels")[0], int))


class TestAudioBatches(Test):
    def test_audio_bathes(self):
        kaldir = self.get_data_path("kaldi/complete")
        with open(kaldir + "/text") as f:
            kaldir_ids = [l.split()[0] for l in f.readlines()]
        files = [self.get_data_path(f) for f in ["audio/bonjour.wav", "audio/cfpp2channels.mp3", "audio/bonjour.wav"]]

        format_to_type = {
            "torch": torch.Tensor,
            "array": np.ndarray,
            "bytes": bytes,
        }

        for source, num_segments, expected_ids in [
            (kaldir, len(kaldir_ids), kaldir_ids),
            (files, len(files), [os.path.basename(f) for f in files]),
            (np.array([0, 0, 0]), 1, None),
        ]:
            for format in "torch", "array", "bytes":
                for batch_size in (
                    0,
                    1,
                    2,
                ):
                    for output_ids in False, True:
                        res = list(
                            to_audio_batches(
                                source,
                                batch_size=batch_size,
                                return_format=format,
                                output_ids=output_ids,
                            )
                        )

                        if batch_size != 0:
                            if len(res) > 1:
                                self.assertTrue(min([len(b) == batch_size for b in res[:-1]]))
                            else:
                                self.assertGreater(len(res[0]), 0)
                            res = flatten(res)
                        self.assertEqual(len(res), num_segments)
                        if output_ids:
                            self.assertTrue(all([isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], str) for r in res]))
                            if expected_ids:
                                ids = [r[1] for r in res]
                                self.assertEqual(ids, expected_ids)
                            res = [r[0] for r in res]
                        self.assertTrue(all([isinstance(r, format_to_type[format]) for r in res]))
