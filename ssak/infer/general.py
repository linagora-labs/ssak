from enum import Enum

import speechbrain as sb
import torch

from ssak.infer.speechbrain_infer import (
    _speechbrain_classes,
    get_tokenizer_vocab,
    speechbrain_compute_logits,
    speechbrain_infer,
    speechbrain_load_model,
)
from ssak.infer.torchaudio_infer import (
    torchaudio_compute_logits,
    torchaudio_is_valid,
    torchaudio_load_model,
)
from ssak.infer.transformers_infer import (
    WAV2VEC_CLASSES,
    transformers_compute_logits,
    transformers_infer,
    transformers_load_model,
)


class ModelType(str, Enum):
    SPEECHBRAIN = "SpeechBrain"
    TRANSFORMERS = "Transformers"
    TORCHAUDIO = "TorchAudio"
    WHISPER = "Whisper"

    def __str__(self):
        return self.value


def load_model(source, device=None):
    try:
        return speechbrain_load_model(source, device)
    except Exception as e1:
        try:
            return transformers_load_model(source, device)
        except Exception as e2:
            try:
                return torchaudio_load_model(source, device)
            except Exception as e3:
                raise ValueError(f"Unknown model type: {source}:\n{e1}\n{e2}\n{e3}")


def get_model_type(model):
    # if isinstance(model, str):
    #     return get_model_type(load_model(model))

    if isinstance(model, _speechbrain_classes):
        return ModelType.SPEECHBRAIN

    elif isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], WAV2VEC_CLASSES):
        return ModelType.TRANSFORMERS

    elif torchaudio_is_valid(model):
        return ModelType.TORCHAUDIO

    raise NotImplementedError(f"Unknown model type: {type(model)}")


def infer(model, batch, **kwargs):
    if isinstance(model, str):
        return infer(load_model(model), batch, **kwargs)

    model_type = get_model_type(model)

    if model_type == ModelType.SPEECHBRAIN:
        return speechbrain_infer(model, batch, **kwargs)

    elif model_type == ModelType.TRANSFORMERS:
        return transformers_infer(model, batch, **kwargs)

    else:
        raise NotImplementedError(f"infer() not implemented for model type: {type(model)}")


def compute_logits(model, batch):
    if isinstance(model, str):
        return compute_log_probas(load_model(model), batch)

    model_type = get_model_type(model)

    if model_type == ModelType.SPEECHBRAIN:
        reco, logits = speechbrain_compute_logits(model, batch)

    elif model_type == ModelType.TRANSFORMERS:
        model, processor = model
        logits = transformers_compute_logits(model, processor, batch)
        logits = logits[0, :, :]

    elif model_type == ModelType.TORCHAUDIO:
        logits = torchaudio_compute_logits(model, batch)

    else:
        raise NotImplementedError(f"compute_log_probas() not implemented for model type: {model_type}")

    return logits


def compute_log_probas(model, batch):
    logits = compute_logits(model, batch)
    return torch.log_softmax(logits, dim=-1)  # .cpu().numpy()


def decode_log_probas(model, logits):
    if isinstance(model, str):
        return decode_log_probas(load_model(model), logits)

    model_type = get_model_type(model)

    if model_type == ModelType.SPEECHBRAIN:
        _, blank_id = get_model_vocab(model)
        indices = sb.decoders.ctc_greedy_decode(logits.unsqueeze(0), torch.Tensor([1.0]), blank_id=blank_id)
        reco = model.tokenizer.decode(indices)
        return reco[0]

    elif model_type == ModelType.TRANSFORMERS:
        model, processor = model
        return processor.decode(torch.argmax(logits, dim=-1))

    else:
        raise NotImplementedError(f"decode_log_probas() not implement for model type: {model_type}")


def get_model_vocab(model):
    if isinstance(model, str):
        return get_model_vocab(load_model(model))

    model_type = get_model_type(model)

    if model_type == ModelType.SPEECHBRAIN:
        return get_tokenizer_vocab(model.tokenizer)

    elif model_type == ModelType.TRANSFORMERS:
        processor = model[1]
        labels_dict = dict((v, k) for k, v in processor.tokenizer.get_vocab().items())
        labels = [labels_dict[i] for i in range(len(labels_dict))]
        labels = [l if l != "|" else " " for l in labels]
        blank_id = labels.index("<pad>") if "<pad>" in labels else labels.index("[PAD]") if "[PAD]" in labels else -1
        if blank_id == -1:
            raise ValueError("Neither <pad> nor [PAD] found in labels")
        return labels, blank_id

    elif model_type == ModelType.TORCHAUDIO:
        _, labels = model
        labels = list(labels)
        # blank_id = labels.index("-") # ...? Is it general enough?
        blank_id = 0
        labels = [l if l != "|" else " " for l in labels]
        return labels, blank_id

    else:
        raise NotImplementedError(f"get_model_vocab() not implemented for model type: {model_type}")


def get_model_sample_rate(model):
    if isinstance(model, str):
        return get_model_sample_rate(load_model(model))

    model_type = get_model_type(model)

    if model_type == ModelType.SPEECHBRAIN:
        return model.audio_normalizer.sample_rate

    elif model_type == ModelType.TRANSFORMERS:
        processor = model[1]
        return processor.feature_extractor.sampling_rate

    elif model_type == ModelType.TORCHAUDIO:
        # Can we do better?
        return 16000

    else:
        raise NotImplementedError(f"get_model_sample_rate() not implemented for model type: {model_type}")


### Torch audio models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot the distribution of log-probas on some audio (to check helpers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", help="Input model folder or name (Transformers, Speechbrain)", type=str)
    parser.add_argument("audio", help="Input audio files", type=str, nargs="+")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    from ssak.utils.dataset import to_audio_batches

    model = load_model(args.model)
    all_logits = np.array([])
    for audio in to_audio_batches(args.audio):
        logits = compute_log_probas(model, audio)
        exp_logits = np.exp(logits)
        sum_per_frame = exp_logits.sum(axis=-1)
        print("min/max sum per frame:", min(sum_per_frame), max(sum_per_frame))
        # Flatten the logits
        logits = logits.reshape(-1)
        all_logits = np.concatenate((all_logits, logits))

    plt.hist(all_logits, bins=100, range=(-25, 1))
    plt.show()
