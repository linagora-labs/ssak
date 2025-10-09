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

def get_available_contexts_categories():
    return list(CONTEXTS.keys())

def get_contexts(category):
    return CONTEXTS.get(category, "default_contexts")