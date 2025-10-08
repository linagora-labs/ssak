import argparse
import json
import logging
import os
import re
import random
import shutil

import numpy as np
from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONTEXTS = {
    "youtube_contexts":[
        "Écris les sous-titres de l'audio en français.",
        "Génère une transcription adaptée aux sous-titres YouTube en français.",
        "Fournis les dialogues synchronisables pour une vidéo YouTube en français.",
        "Crée des sous-titres clairs et concis pour cette vidéo en français.",
        "Transcris le contenu parlé comme pour une publication YouTube en français."
    ],
    "cleaned_contexts":[
        "Transcrivez l'audio en français suivant de la manière la plus propre possible.",
        "Fournissez une version épurée et fluide de la transcription en français.",
        "Transcrivez l'audio en corrigeant les erreurs de langage oral et en améliorant la lisibilité.",
        "Rédigez une transcription claire, sans hésitations ni répétitions inutiles.",
        "Transformez l'audio en un texte français fluide et bien structuré."
    ],
    "conversation_contexts":[ 
        "Transcrivez la conversation en français en omettant le bruit.",
        "Fournissez une transcription propre de cette conversation en français.",
        "Transcrivez fidèlement le dialogue entre les participants en français.",
        "Produisez une transcription claire de la discussion, sans éléments non verbaux.",
        "Rédigez une version lisible de cet échange oral en français."
    ],
    "noisy_contexts":[
        "Transcrivez la conversation en français en omettant le bruit.",
        "Fournissez une transcription propre de cette conversation en français.",
        "Transcrivez fidèlement le dialogue entre les participants en français.",
        "Produisez une transcription claire de la discussion, sans éléments non verbaux.",
        "Rédigez une version lisible de cet échange oral en français."
    ],
    "speech_contexts":[
        "Transcrivez le discours en français en omettant les hésitations.",
        "Transcrivez le discours en français en respectant la ponctuation pour plus de clarté.",
        "Générez une transcription écrite de ce discours en français, en vous assurant qu'aucun détail n'est omis.",
        "Fournissez une transcription formelle et structurée de cette allocution en français.",
        "Rédigez le texte complet de ce discours comme s'il était destiné à une publication écrite."
    ],
    "default_contexts":[
        "Transcrivez l'audio en français suivant de la manière la plus précise possible.",
        "Écrivez exactement ce qui est dit dans cet audio en français.",
        "Fournissez une transcription complète du discours en français dans ce clip audio.",
        "Écoutez l'audio en français et transcrivez-le mot pour mot.",
        "Écrivez la transcription complète de ce texte parlé en français.",
        "Transcrivez fidèlement tout ce qui est audible dans cet enregistrement en français.",
        "Faites une transcription intégrale du contenu audio en français."
    ]
}

def set_context(input_file, output_file, contexts):
    dataset = NemoDataset()
    dataset.load(input_file)
    for row in tqdm(dataset, desc=f"Setting context on {input_file}"):
        row.context = random.choice(contexts)
    dataset.save(output_file+".tmp")
    shutil.move(output_file+".tmp", output_file)

def set_context_on_folder(folder, context_file):
    with open(context_file, "r") as f:
        contexts_dict = json.load(f)
    for root, dirs, files in tqdm(os.walk(folder)):
        for file in files:
            dataset_name = file.split("_")[1:]
            if dataset_name not in contexts_dict:
                logger.info(f"Dataset {dataset_name} has no context defined, using default contexts")
                contexts = CONTEXTS["default_contexts"].copy()
            else: 
                contexts = contexts_dict[dataset_name]
                if not isinstance(contexts, list):
                    contexts = CONTEXTS[contexts]
            set_context(os.path.join(root, file), os.path.join(root, file.replace(".json", "_context.json")), contexts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines by looking at the number of words and segment duration from nemo manifest")
    parser.add_argument("folder", help="Input folder", type=str)
    parser.add_argument("context_file", help="context_file", type=str)
    args = parser.parse_args()
    set_context_on_folder(args.folder, args.context_file)