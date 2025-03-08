import os
import re
import shutil

from .utils import Test


class TestTrainSpeechbrain(Test):
    def setUp(self):
        super().setUp()
        # Set HOME environment to root, to check it even works in this setting
        os.environ["HOME"] = "/"

    def get_sb_path(self, fname):
        return self.get_lib_path("train/speechbrain/" + fname)

    def opts(self):
        return [
            "--train",
            self.get_data_path("kaldi/train_weighted.txt"),
            "--valid",
            self.get_data_path("kaldi/minimal"),
            "--batch_size",
            "4",
            "--num_epochs",
            "2",
            "--max_duration",
            "10",
        ]

    def test_train_speechbrain_fromscratch(self):
        hparams_file = "hyperparameters_wav2vec_fromscratch.yaml"
        dir = self.get_output_path("speechbrain_LeBenchmark-wav2vec2-FR-7K-large_len-1-10_frTrue_lr1.0-0.0001_bs4_s1234_ascending")
        shutil.rmtree(dir, ignore_errors=True)

        self.assertRun(
            [
                self.get_sb_path("wav2vec_train.py"),
                self.get_sb_path("fr/" + hparams_file),
                "--freeze_wav2vec",
                "True",
                *self.opts(),
            ]
        )

        self.assertFolderContentIsRight(dir, "fromscratch", "j", hparams_file)

    def test_train_speechbrain_finetune(self):
        hparams_file = "hyperparameters_wav2vec_finetune_cv-fr.yaml"
        dir = self.get_output_path("speechbrain_asr-wav2vec2-commonvoice-fr_len-0.5-10_frTrue_lr1.0-0.0001_bs4_s1234_random")
        shutil.rmtree(dir, ignore_errors=True)

        self.assertRun(
            [
                self.get_sb_path("wav2vec_train.py"),
                self.get_sb_path("fr/" + hparams_file),
                *self.opts(),
            ]
        )

        self.assertFolderContentIsRight(dir, "finetuning", "BONJ", hparams_file)

    def assertFolderContentIsRight(self, dir, name, expected, hparams_file):
        self.assertTrue(os.path.isdir(dir))
        self.assertTrue(os.path.isdir(dir + "/src"))
        self.assertTrue(os.path.isfile(dir + "/src/" + hparams_file))
        self.assertTrue(os.path.isfile(dir + "/src/train_weighted.txt"))
        self.assertTrue(not os.path.exists(dir + "/minimal"))
        self.assertTrue(os.path.isdir(dir + "/train_log"))

        self.assertNonRegression(
            dir + "/train_log.txt",
            f"train_speechrain/train_log_{name}.txt",
            lambda line: re.sub(r"_time_h: [0-9e\.\-\+]+", "train_time_h: XXX", line),
        )

        # Check finalization
        final_dir = dir + "/final"
        self.assertTrue(final_dir)

        stdout = self.assertRun(
            [
                self.get_lib_path("infer/speechbrain_infer.py"),
                self.get_data_path("audio/bonjour.wav"),
                "--model",
                final_dir,
            ]
        )
        self.assertEqual(stdout, expected + "\n")

        shutil.rmtree(dir)
