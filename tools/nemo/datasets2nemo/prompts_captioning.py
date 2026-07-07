"""Shared instruction pools for sound-captioning manifests.

Caption length varies a lot by dataset (AudioCaps / WavCaps average ~8 words, MECAT
~13), so the instruction has to match the target length — otherwise the model gets a
contradictory signal (a "short caption" prompt paired with a long, detailed target).

Pools:
    NEUTRAL_PROMPTS  - no length cue either way.
    SHORT_PROMPTS    - explicitly ask for a short/brief caption.
    LONG_PROMPTS     - explicitly ask for a detailed caption; use ONLY for datasets that
                       actually carry long captions (e.g. the MECAT "long" captions).

Wording is kept consistent with the MECAT converter
(datasets/audio/nemo/_script_/python/get_data/get_mecat_caption.py) so the short-vs-long
instruction signal is shared across datasets.

SHORT_CAPTION_PROMPTS = NEUTRAL + SHORT is the pool for datasets whose captions are short
single phrases (AudioCaps, WavCaps): it mixes neutral asks with explicit "short" asks and
never asks for detail.
"""

# Neutral asks — no length cue.
NEUTRAL_PROMPTS = [
    "Describe what you hear in the audio.",
    "What sounds can you hear in this recording?",
    "Caption the audio: what is happening?",
    "Tell me what this sound is.",
    "Summarize the sound events present in the clip.",
    "What is happening in this audio?",
    "Describe the acoustic scene.",
    "What do you hear in this audio excerpt?",
    "Write a caption for this audio clip.",
    "Identify the sounds in this clip.",
    "What is being heard in the recording?",
    "Please generate an audio caption for the clip.",
    "Describe the sounds in the recording.",
    "Provide a caption for this audio excerpt."
]

# Explicitly ask for a short/brief caption (matches MECAT's `short` instruction pool).
SHORT_PROMPTS = [
    "Briefly describe what you hear.",
    "Give a short description of the audio excerpt.",
    "Summarize what you hear.",
    "Describe the sounds of the audio clip briefly.",
    "Provide a brief audio description.",
    "Shortly describe what you hear.",
    "Give a concise description of the audio excerpt.",
    "Describe what is heard in a few words.",
    "Provide a short summary of the sounds.",
    "Quickly describe what you hear.",
    "In a few words, what do you hear?",
    "Write a short caption for this audio.",
    "Can you give a short caption of this clip?"
]

# Explicitly ask for a detailed caption — ONLY for long-caption datasets (e.g. MECAT-long).
LONG_PROMPTS = [
    "Describe in detail what you hear.",
    "Give a detailed description of the audio.",
    "Explain everything you can hear in the audio.",
    "Provide a full description of the sounds you hear.",
    "Describe all audible elements in detail.",
    "Give a comprehensive description of the audio content.",
    "Describe the audio scene as thoroughly as possible.",
    "Explain what is happening in the audio in detail.",
    "Provide a detailed account of what you hear.",
    "Describe the sounds and events in the audio in depth.",
]

# Pool for short-caption datasets (AudioCaps, WavCaps): neutral + short, never long.
SHORT_CAPTION_PROMPTS = NEUTRAL_PROMPTS + SHORT_PROMPTS
