"""Shared instruction pools for sound-captioning manifests.

Pools:
    NEUTRAL_PROMPTS  - no length cue either way.
    SHORT_PROMPTS    - explicitly ask for a short/brief caption.
    LONG_PROMPTS     - explicitly ask for a detailed caption; use ONLY for datasets that
                       actually carry long captions (e.g. the MECAT "long" captions).
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

# ----------------------------------------------------------------------------
# French pools.
#
# Authored, not machine-translated: MT renders "clip" as "vidéo"/"séquence vidéo",
# which tells an audio-only model it is watching video. "clip" is always "extrait".
# ----------------------------------------------------------------------------

NEUTRAL_PROMPTS_FR = [
    "Décrivez ce que vous entendez dans l'extrait audio.",
    "Quels sons entendez-vous dans cet enregistrement ?",
    "Légendez l'extrait audio : que se passe-t-il ?",
    "Dites-moi ce qu'est ce son.",
    "Résumez les événements sonores présents dans l'extrait.",
    "Que se passe-t-il dans cet extrait audio (bruits, musiques, événements, etc.) ?",
    "Décrivez la scène sonore.",
    "Qu'entendez-vous dans cet extrait audio ?",
    "Rédigez une légende pour cet extrait audio.",
    "Identifiez les sons présents dans cet extrait.",
    "Qu'entend-on dans cet enregistrement ?",
    "Générez une légende audio pour cet extrait.",
    "Décrivez les sons présents dans l'enregistrement.",
    "Proposez une légende pour cet extrait audio.",
]

SHORT_PROMPTS_FR = [
    "Décrivez brièvement ce que vous entendez.",
    "Donnez une courte description de l'extrait audio.",
    "Résumez en quelques mots les sons et bruits que vous entendez.",
    "Décrivez brièvement les sons de l'extrait audio.",
    "Fournissez une brève description audio.",
    "Décrivez sommairement ce que vous entendez.",
    "Donnez une description concise de l'extrait audio.",
    "Décrivez en quelques mots ce que l'on entend.",
    "Fournissez un bref résumé des sons.",
    "Décrivez rapidement ce que vous entendez.",
    "En quelques mots, qu'entendez-vous comme sons, bruits, musiques, etc. ?",
    "Rédigez une courte légende pour cet extrait audio.",
    "Donnez une courte légende de cet extrait.",
]

LONG_PROMPTS_FR = [
    "Décrivez en détail ce que vous entendez.",
    "Donnez une description détaillée de l'extrait audio.",
    "Expliquez tout ce que vous pouvez entendre dans l'extrait audio.",
    "Fournissez une description complète des sons que vous entendez.",
    "Décrivez en détail tous les éléments audibles.",
    "Donnez une description exhaustive du contenu audio.",
    "Décrivez la scène sonore aussi précisément que possible.",
    "Expliquez en détail ce qui se passe dans l'extrait audio.",
    "Fournissez un compte rendu détaillé de ce que vous entendez.",
    "Décrivez en profondeur les sons et les événements de l'extrait audio.",
]

SHORT_CAPTION_PROMPTS_FR = NEUTRAL_PROMPTS_FR + SHORT_PROMPTS_FR

# English prompt -> French prompt, for rewriting prompts in already-generated
# French manifests (built by translating the English ones).
EN_TO_FR_PROMPT = dict(
    zip(NEUTRAL_PROMPTS + SHORT_PROMPTS + LONG_PROMPTS,
        NEUTRAL_PROMPTS_FR + SHORT_PROMPTS_FR + LONG_PROMPTS_FR)
)
