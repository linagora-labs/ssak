import os
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from ssak.infer.kaldi_infer import *
from ssak.infer.whisper_infer import *


def load_audio(file_path):
    audio, sr = librosa.load(file_path)
    return audio, sr


def visualize_audio(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.linspace(0, len(audio) / sr, num=len(audio)), audio)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    st.pyplot(fig)


def render_transcription(transcription, border_style):
    st.header("Transcription")
    if isinstance(transcription, str):
        st.markdown(border_style, unsafe_allow_html=True)
        st.markdown(f'<ul class="transcription-list">{transcription}</ul>', unsafe_allow_html=True)
    else:
        st.markdown(f'<ul class="transcription-list">{transcription}</ul>', unsafe_allow_html=True)


def get_language_border_style(language):
    direction = "rtl" if language == "ar" else "ltr"
    return f"""
    <style>
    .transcription-list {{
        border: 1px solid #000;
        padding: 10px;
        border-radius: 5px;
        direction: {direction};
        text-align: {direction};
    }}
    </style>
    """


kaldi_m = st.sidebar.checkbox("KALDI TN")


def main():
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }

        .transcription {
            border: 'bold','1px';
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 16px;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Transcriberlit")
    st.write("Welcome to my Transcriberlit. This app allows you to transcribe audio files or audio clips using the Whisper library.")
    st.header("Instructions")
    st.write('To use this app, upload an audio file or record a clip using the microphone. Then click the "Transcribe Audio" button to transcribe the audio.')

    sample_rate = 16_000

    models = {
        "TN_model": "/home/usertn2/Documents/Data/Models/TN_model",
        "Tiny": "tiny",
        "Base": "base",
        "Small": "small",
    }

    Language = {
        "Arabic": "ar",
        "English": "en",
        "French": "fr",
    }

    model_name = None

    if kaldi_m:
        model_name = "/home/usertn2/Documents/Data/Models/LinSTT_TN_ASR_from_scratch_20340203/am,/home/usertn2/Documents/Data/Models/LinSTT_TN_ASR_from_scratch_20340203/graph"
    else:
        selected_model = st.sidebar.selectbox("Select a model", list(models.keys()), index=0)
        model_name = models[selected_model]

    language = st.sidebar.selectbox("Select a language", list(Language.keys()), index=0)
    language = Language[language]

    border_style = get_language_border_style(language)

    audio_file = st.file_uploader("Upload an audio File:", type=["wav", "mp3", "flac"])

    audio_bytes = audio_recorder(
        energy_threshold=(-2.0, 2.0),
        pause_threshold=15.0,
        sample_rate=sample_rate,
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x",
    )

    # If the "Transcribe Audio" button is clicked, transcribe the audio
    if st.sidebar.button("Transcribe Audio"):
        if audio_file is not None:
            # Create a temporary directory to store the file
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = os.path.join(temp_dir.name, audio_file.name)

            # Save the file to the temporary location
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())

            audio, sr = load_audio(temp_path)
            st.audio(temp_path)
            visualize_audio(audio, sr)
            if not kaldi_m:
                for transcription in whisper_infer(
                    model_name,
                    temp_path,
                    language=language,
                ):
                    render_transcription(transcription, border_style)
            else:
                for transcription in kaldi_infer(
                    model_name,
                    temp_path,
                    batch_size=4,
                ):
                    render_transcription(transcription, border_style)

        elif audio_bytes:
            filename = "audio_record.wav"
            # Create a temporary directory to store the file
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = os.path.join(temp_dir.name, filename)
            with open(temp_path, "wb") as audio_f:
                audio_f.write(audio_bytes)
            audio, sr = load_audio(temp_path)
            with open(temp_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")
                visualize_audio(audio, sr)
                for transcription in whisper_infer(
                    model_name,
                    audio_file.name,
                    batch_size=1,
                    language=language,
                ):
                    st.header("Transcription")
                    render_transcription(transcription, border_style)
        else:
            st.error("You need to record/upload an audio file")


if __name__ == "__main__":
    main()
