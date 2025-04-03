import random

import nltk
import parler_tts
import torch
import torchaudio
import transformers
from nltk.tokenize import sent_tokenize

# Download NLTK's punkt tokenizer data if not already downloaded
global _nltk_initialized
_nltk_initialized = False

# Function to split text into chunks using sentence tokenization
def nltk_chunk_text(text):
    global _nltk_initialized
    if not _nltk_initialized:
        nltk.download('punkt')
        _nltk_initialized = True
    return sent_tokenize(text)

_tts_speaker_prompts = [
    "A female speaker delivers an expressive and animated speech with a very high-pitch voice. "
    "The recording is slightly noisy but of good quality, as her voice comes across as very close-sounding.",
    "A female speaker delivers her speech with a slightly expressive and animated tone, "
    "her voice ringing clearly and undistorted in the recording. "
    "The pitch of her voice is very high, adding a sense of urgency and excitement.",
    "A female speaks with a slightly expressive and animated tone in a recording that sounds quite clear and close up. "
    "There is only a mild amount of background noise present, and her voice has a moderate pitch. "
    "Her speech pace is steady, neither slow nor particularly fast.",
    "A female speaker delivers her speech in a recording that sounds clear and close up. "
    "Her voice is slightly expressive and animated, with a moderate pitch. "
    "The recording has a mild amount of background noise, but her voice is still easily understood.",
    "In a somewhat confined space, a female speaker delivers a talk that is slightly expressive and animated, "
    "despite some background noise. "
    "Her voice has a low-pitch tone.",
    "A male voice speaks in a monotone tone with a slightly low-pitch, delivering his words at a moderate speed. "
    "The recording offers almost no noise, resulting in a very clear and high-quality listen. "
    "The close-up microphone captures every detail of his speech.",
    "A man speaks with a monotone tone and a slightly low-pitch, delivering his words at a moderate speed. "
    "The recording captures his speech very clearly and distinctly, with little to no background noise. "
    "The listener feels as if they're almost sharing the same space with the speaker.",
    "A male speaker delivers his words with a very monotone and slightly faster than average pace. "
    "His voice is very clear, making every word distinct, while it also has a slightly low-pitch tone. "
    "The recording quality is excellent, with no apparent reverberation or background noise.",
    "A male speaker delivers his words in a very monotone and slightly low-pitched voice, "
    "maintaining a moderate speed. The recording is of very high quality, with minimum noise "
    "and a very close-sounding reverberation that suggests a quiet and enclosed environment.",
]


global _tts_models
_tts_models = {}


def text_to_speech(
    text,
    prompt=None,
    device=None,
    model_name="parler-tts/parler-tts-mini-multilingual-v1.1",
    sampling_rate=16_000,
):
    global _tts_models

    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if prompt is None:
        prompt = random.choice(_tts_speaker_prompts)
    elif isinstance(prompt, list):
        prompt = random.choice(prompt)
    elif isinstance(prompt, str):
        pass
    else:
        raise ValueError("Prompt must be a string or a list of strings")

    # Load processor and model from Hugging Face, with caching in (V)RAM
    if model_name not in _tts_models:

        model = parler_tts.ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = transformers.AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        model_sampling_rate = model.config.sampling_rate

        _tts_models[model_name] = (model, tokenizer, description_tokenizer, model_sampling_rate)

    (model, tokenizer, description_tokenizer, model_sampling_rate) = _tts_models[model_name]
    model = model.to(device)

    text_tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
    speaker_type_prompt_tokens = description_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    audio_tensor = model.generate(input_ids=speaker_type_prompt_tokens, prompt_input_ids=text_tokens)

    if len(audio_tensor.shape) == 2 and audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor[0]

    audio_tensor = audio_tensor.to("cpu")

    if sampling_rate != model_sampling_rate:
        audio_tensor = torchaudio.transforms.Resample(model_sampling_rate, sampling_rate)(audio_tensor)

    audio_tensor = audio_tensor.numpy()

    return audio_tensor

if __name__ == "__main__":

    import argparse
    import os

    from audio import save_audio
    parser = argparse.ArgumentParser()
    parser.add_argument("words", type=str, nargs="+", help="Text to convert to speech")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference")
    parser.add_argument("--model_name", type=str, default="parler-tts/parler-tts-mini-multilingual-v1.1",
        help="Model name or path")
    parser.add_argument("--output", type=str, default="out", help="Output folder name")
    parser.add_argument("--num", type=int, default=10, help="Number of generations")
    args = parser.parse_args()

    text = " ".join(args.words)

    for i in range(args.num):
        prompt = random.choice(_tts_speaker_prompts)
        audio_tensor = text_to_speech(text, prompt, model_name=args.model_name, device=args.device)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, f"audio_{i:03d}_prompt.txt"), "w") as f:
            f.write(prompt)
        save_audio(os.path.join(args.output, f"audio_{i:03d}.wav"), audio_tensor)

