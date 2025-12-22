import json
import os
import re

# Output folder
output_folder = "translation_prompts"
os.makedirs(output_folder, exist_ok=True)

# Language codes
language_codes = {
    "French": "fr",
    "English": "en",
    "Spanish": "es",
    "German": "de",
    "Italian": "it"
}

# Localized language names
localized_languages = {
    "French": {"French": "français", "English": "anglais", "Spanish": "espagnol", "German": "allemand", "Italian": "italien"},
    "English": {"French": "French", "English": "English", "Spanish": "Spanish", "German": "German", "Italian": "Italian"},
    "Spanish": {"French": "francés", "English": "inglés", "Spanish": "español", "German": "alemán", "Italian": "italiano"},
    "German": {"French": "Französisch", "English": "Englisch", "Spanish": "Spanisch", "German": "Deutsch", "Italian": "Italienisch"},
    "Italian": {"French": "francese", "English": "inglese", "Spanish": "spagnolo", "German": "tedesco", "Italian": "italiano"}
}

# Prompt templates (multiple variations)
templates = {
    "full": {
        "French": [
            "Traduisez le contenu suivant du {lang1} vers le {lang2}.",
            "Veuillez traduire cet enregistrement du {lang1} en {lang2}.",
            "Peux-tu traduire ce discours du {lang1} vers le {lang2} ?",
            "Convertissez le contenu parlé du {lang1} en {lang2}.",
            "Merci de traduire ce fichier audio du {lang1} en {lang2}."
        ],
        "English": [
            "Translate the following content from {lang1} to {lang2}.",
            "Please translate this audio recording from {lang1} into {lang2}.",
            "Convert the spoken content from {lang1} to {lang2}.",
            "Translate the speech provided from {lang1} to {lang2}.",
            "Translate this file from {lang1} into {lang2}."
        ],
        "Spanish": [
            "Traduce el siguiente contenido de {lang1} a {lang2}.",
            "Por favor, traduzca esta grabación de {lang1} a {lang2}.",
            "Convierte el contenido hablado de {lang1} a {lang2}.",
            "Traduce el discurso proporcionado de {lang1} a {lang2}.",
            "Traduce este archivo de {lang1} a {lang2}."
        ],
        "German": [
            "Übersetzen Sie den folgenden Inhalt von {lang1} nach {lang2}.",
            "Bitte übersetzen Sie diese Audioaufnahme von {lang1} ins {lang2}e.",
            "Konvertieren Sie den gesprochenen Inhalt von {lang1} nach {lang2}.",
            "Übersetzen Sie die bereitgestellte Rede von {lang1} nach {lang2}.",
            "Bitte übersetzen Sie diese Datei von {lang1} ins {lang2}e."
        ],
        "Italian": [
            "Traduci il seguente contenuto dal {lang1} al {lang2}.",
            "Per favore, traduci questa registrazione audio dal {lang1} al {lang2}.",
            "Converti il contenuto parlato dal {lang1} al {lang2}.",
            "Traduci il discorso fornito dal {lang1} al {lang2}.",
            "Traduci questo file dal {lang1} al {lang2}."
        ]
    },
    "target_only": {
        "French": [
            "Traduis le contenu suivant en {lang2}.",
            "Veuillez traduire cet enregistrement en {lang2}.",
            "Peux-tu traduire ce discours en {lang2} ?",
            "Convertissez le contenu parlé en {lang2}.",
            "Merci de traduire ce fichier audio en {lang2}."
        ],
        "English": [
            "Translate the following content into {lang2}.",
            "Please translate this audio recording into {lang2}.",
            "Convert the spoken content to {lang2}.",
            "Translate the provided speech into {lang2}.",
            "Translate this file into {lang2}."
        ],
        "Spanish": [
            "Traduce el siguiente contenido a {lang2}.",
            "Por favor, traduzca esta grabación a {lang2}.",
            "Convierte el contenido hablado a {lang2}.",
            "Traduce el discurso proporcionado a {lang2}.",
            "Traduce este archivo a {lang2}."
        ],
        "German": [
            "Übersetzen Sie den folgenden Inhalt ins {lang2}e.",
            "Bitte übersetzen Sie diese Audioaufnahme ins {lang2}e.",
            "Konvertieren Sie den gesprochenen Inhalt ins {lang2}e.",
            "Übersetzen Sie die bereitgestellte Rede ins {lang2}e.",
            "Bitte übersetzen Sie diese Datei ins {lang2}e."
        ],
        "Italian": [
            "Traduci il seguente contenuto in {lang2}.",
            "Per favore, traduci questa registrazione audio in {lang2}.",
            "Converti il contenuto parlato in {lang2}.",
            "Traduci il discorso fornito in {lang2}.",
            "Traduci questo file in {lang2}."
        ]
    },
    "implicit_only": {
        "French": ["Traduisez", "Convertissez le contenu", "Traduis"],
        "English": ["Translate", "Convert the content", "Translate what is being said"],
        "Spanish": ["Traduce", "Convierte el contenido", "Traduce lo que se dice"],
        "German": ["Übersetzen Sie", "Konvertieren Sie den Inhalt", "Übersetzen Sie das Gesagte"],
        "Italian": ["Traduci", "Converti il contenuto", "Traduci ciò che viene detto"]
    }
}



languages = ["French", "English", "Spanish", "German", "Italian"]

def fix_prompt(text):
    text = re.sub(r"\bdu\s+([aeiouhAEIOUH])", r"de l'\1", text)
    text = re.sub(r"\ble\s+([aeiouhAEIOUH])", r"l'\1", text)
    # it
    text = re.sub(r'\bdallo\s+([aeiouAEIOU])', r"dall'\1", text)
    # da + s+consonant → dallo
    text = re.sub(r'\bdalo\s+([sS][bcdfghjklmnpqrstvwxyz])', r'dallo \1', text)
    
    # a + vowel → all'
    text = re.sub(r'\bal\s+([aeiouAEIOU])', r"all'\1", text)
    # a + s+consonant → allo
    text = re.sub(r'\bal\s+([sS][bcdfghjklmnpqrstvwxyz])', r'allo \1', text)
    return text

# Generate JSON files
for src_lang in languages:
    for tgt_lang in languages:
        if src_lang == tgt_lang:
            continue

        prompts = {
            0.75: [],
            0.25: []
        }

        # 1. Full prompts
        for prompt_lang in languages:
            for template in templates["full"][prompt_lang]:
                p = template.format(
                    lang1=localized_languages[prompt_lang][src_lang],
                    lang2=localized_languages[prompt_lang][tgt_lang]
                )
                p = fix_prompt(p)
                if prompt_lang==src_lang or prompt_lang==tgt_lang:
                    prompts[0.75].append(p)
                else:
                    prompts[0.25].append(p)

        # Target-only prompts
        for prompt_lang in languages:
            for template in templates["target_only"][prompt_lang]:
                p = template.format(
                    lang2=localized_languages[prompt_lang][tgt_lang]
                )
                p = fix_prompt(p)
                if prompt_lang==src_lang or prompt_lang==tgt_lang:
                    prompts[0.75].append(p)
                else:
                    prompts[0.25].append(p)
        # 3
        for template in templates["implicit_only"][tgt_lang]:
            prompts[0.75].append(template)

        # Save JSON file
        data = {"default_contexts": prompts}
        file_name = f"{language_codes[src_lang]}-{language_codes[tgt_lang]}_ast_contexts.json"
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Generated {file_path}")
