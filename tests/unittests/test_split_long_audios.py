import shutil

from .utils import Test


class TestSplitLongAudios(Test):
    def test_split_long_audios_speechbrain(self):
        output_folder = self.get_temp_path("split_long_audio_speechbrain")
        shutil.rmtree(output_folder, ignore_errors=True)
        self.assertRun(
            [
                self.get_tool_path("align_audio_transcript.py"),
                self.get_data_path("kaldi/small"),
                output_folder,
                "--model",
                "speechbrain/asr-wav2vec2-commonvoice-fr",
                "--max_duration",
                "4",
            ]
        )
        self.assertNonRegression(output_folder, "align_audio_transcript/speechbrain")
        shutil.rmtree(output_folder)

    def test_split_long_audios_transformers(self):
        output_folder = self.get_temp_path("split_long_audio_transformers")
        shutil.rmtree(output_folder, ignore_errors=True)
        self.assertRun(
            [
                self.get_tool_path("align_audio_transcript.py"),
                self.get_data_path("kaldi/small"),
                output_folder,
                "--model",
                "Ilyes/wav2vec2-large-xlsr-53-french",
                "--max_duration",
                "4",
            ]
        )
        self.assertNonRegression(output_folder, "align_audio_transcript/transformers")
        shutil.rmtree(output_folder)

    def test_split_long_audios_torchaudio(self):
        output_folder = self.get_temp_path("split_long_audio_torchaudio")
        shutil.rmtree(output_folder, ignore_errors=True)
        self.assertRun(
            [
                self.get_tool_path("align_audio_transcript.py"),
                self.get_data_path("kaldi/small"),
                output_folder,
                "--model",
                "VOXPOPULI_ASR_BASE_10K_FR",
                "--max_duration",
                "4",
            ]
        )
        self.assertNonRegression(output_folder, "align_audio_transcript/torchaudio")
        shutil.rmtree(output_folder)
