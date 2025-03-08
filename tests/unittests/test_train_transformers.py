import os
import re
import shutil

from .utils import Test


class TestTrainTransformers(Test):
    def setUp(self):
        super().setUp()
        # Set HOME environment to root, to check it even works in this setting
        os.environ["HOME"] = "/"

    def remove_times(self, line):
        line = re.sub(r'_runtime": [0-9e\.\-\+]+', '_runtime": XXX', line)
        line = re.sub(r'_per_second": [0-9e\.\-\+]+', '_per_second": XXX', line)
        line = re.sub(r'_flos": [0-9e\.\-\+]+', '_flos": XXX', line)
        return line

    def test_train_transformers(self):
        dir0 = self.get_output_path("hf_c98d67d4cf540fcdea3d919fb4c4a479_md-10_md-1_bm-Ilyes_wav2vec2-large-xlsr-53-french")
        dir = dir0 + "_lr-0.0001_bs-4_wd-0_ad-0.1_hd-0.05_fpd-0_ld-0.1_mtp-0.05_s-69_adamwt"

        shutil.rmtree(dir0, ignore_errors=True)
        shutil.rmtree(dir, ignore_errors=True)

        self.assertRun(
            [
                self.get_lib_path("train/transformers/wav2vec_train.py"),
                self.get_data_path("kaldi/train_weighted.txt"),
                self.get_data_path("kaldi/minimal"),
                "--batch_size",
                "4",
                "--num_epochs",
                "1",
                "--eval_steps",
                "1",
                "--max_duration",
                "10",
            ]
        )

        self.assertTrue(os.path.isdir(dir0))
        self.assertTrue(os.path.isdir(dir))
        self.assertTrue(os.path.isdir(dir + "/src"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-1"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-2"))
        self.assertFalse(os.path.isdir(dir + "/checkpoint-3"))
        json_file = dir + "/init_eval.json"
        self.assertNonRegression(json_file, "train_transformers/init_eval.json", self.remove_times)
        json_file = dir + "/checkpoint-2/trainer_state.json"
        self.assertNonRegression(json_file, "train_transformers/trainer_state.json", self.remove_times)

        shutil.rmtree(dir0)
        shutil.rmtree(dir)

    def test_train_transformers_with_data_augmentation(self):
        dir0 = self.get_output_path("hf_c98d67d4cf540fcdea3d919fb4c4a479_md-10_md-1_bm-Ilyes_wav2vec2-large-xlsr-53-french")
        dir = dir0 + "_lr-0.0001_bs-4_wd-0_ad-0.1_hd-0.05_fpd-0_ld-0.1_mtp-0.05_s-69_adamwt_augment_online"

        shutil.rmtree(dir0, ignore_errors=True)
        shutil.rmtree(dir, ignore_errors=True)

        self.assertRun(
            [
                self.get_lib_path("train/transformers/wav2vec_train.py"),
                self.get_data_path("kaldi/train_weighted.txt"),
                self.get_data_path("kaldi/minimal"),
                "--batch_size",
                "4",
                "--num_epochs",
                "1",
                "--eval_steps",
                "1",
                "--max_duration",
                "10",
                "--data_augment",
                "--data_augment_noise",
                self.get_data_path("noise"),
                "--data_augment_rir",
                self.get_data_path("[rirs/smallroom/rir_list,rirs/mediumroom/rir_list,rirs/largeroom/rir_list]", check=False),
            ]
        )

        self.assertTrue(os.path.isdir(dir0))
        self.assertTrue(os.path.isdir(dir))
        self.assertTrue(os.path.isdir(dir + "/src"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-1"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-2"))
        self.assertFalse(os.path.isdir(dir + "/checkpoint-3"))
        json_file = dir + "/init_eval.json"
        self.assertNonRegression(json_file, "train_transformers/init_eval.json", self.remove_times)
        json_file = dir + "/checkpoint-2/trainer_state.json"
        self.assertNonRegression(json_file, "train_transformers/trainer_state_augment.json", self.remove_times)

        shutil.rmtree(dir0)
        shutil.rmtree(dir)
