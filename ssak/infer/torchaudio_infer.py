import math

import numpy as np
import torch
import torchaudio

from ssak.utils.env import auto_device  # handles option --gpus


def torchaudio_load_model(source, device=None, download_root="/opt"):
    if device is None:
        device = auto_device()

    bundle = torchaudio.pipelines.__dict__[source]
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    model.eval()
    model.requires_grad_(False)
    return model, labels


def torchaudio_is_valid(model):
    return isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], torchaudio.models.wav2vec2.model.Wav2Vec2Model) and isinstance(model[1], tuple)


def torchaudio_compute_logits(model_and_labels, audios, max_len=2240400):
    # TODO: add support for batch of audios
    # TODO? factorize logic with compute_logits_transformers

    model, _ = model_and_labels

    if len(audios.shape) == 1:
        audios = [audios]

    # Get the device where is running the model
    device = "cpu"
    for p in model.parameters():
        device = p.device
        break

    all_logits = []

    with torch.inference_mode():
        for audio in audios:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            l = len(audio)
            if l > max_len:
                # Split audio in smaller chunks
                print(f"Audio too long, splitting into {math.ceil(l / max_len)} chunks for alignment")
                logits = []
                for i in range(0, l, max_len):
                    j = min(i + max_len, l)
                    logits.append(model(audio[i:j].unsqueeze(0).to(device))[0])
                logits = torch.cat(logits, dim=1)
            else:
                logits, _ = model(audio.unsqueeze(0).to(device))

            all_logits.append(logits.cpu().detach())

    # TODO: support batch of audios
    assert len(all_logits) == 1
    all_logits = all_logits[0]
    if len(all_logits.shape) == 3:
        assert all_logits.shape[0] == 1
        all_logits = all_logits[0, :, :]
    return all_logits
