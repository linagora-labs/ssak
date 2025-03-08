import numpy as np
import torch

COLORS = "bgrcmyk"


def plot_logits(logit, labels_or_model=None, blank_id=None):
    import matplotlib.pyplot as plt

    # Get the labels
    if labels_or_model:
        from ssak.infer.speechbrain_infer import SpeechBrainEncoderASR

        if isinstance(labels_or_model, SpeechBrainEncoderASR):
            tokenizer = labels_or_model.tokenizer
            labels = [tokenizer.decode_ids([i]) for i in range(tokenizer.vocab_size())]
            if blank_id is None:
                blank_id = tokenizer.unk_id()
        elif isinstance(labels_or_model, list):
            labels = labels_or_model
        else:
            raise ValueError(f"Cannot decode labels from {type(labels_or_model)}")
    else:
        labels = list(range(logit.shape[-1]))

    if blank_id is None:
        blank_id = 0

    # Convert to numpy.ndarray
    if isinstance(logit, torch.Tensor):
        logit = logit.detach().cpu().numpy()
    elif isinstance(logit, list):
        logit = np.array(logit)

    # Compute the softmax
    logit = np.exp(logit)
    logit /= logit.sum(axis=-1, keepdims=True)

    plt.figure()
    for i in range(logit.shape[0]):
        plt.subplot(logit.shape[0], 1, i + 1)
        blank_curve = logit[i, :, blank_id]
        other_curve = logit[i, :, list(range(blank_id)) + list(range(blank_id + 1, logit.shape[-1]))].max(-2)
        best_idx = logit[i].argmax(-1)
        assert len(blank_curve) == len(other_curve)
        assert len(blank_curve) == len(best_idx)
        X = np.arange(len(blank_curve))
        plt.bar(X, other_curve + blank_curve, width=1, color="r", alpha=0.2)
        plt.bar(X, other_curve, width=1, color="g")
        plt.ylim(0, 1)
        last = blank_id
        for i, x in enumerate(best_idx):
            if x != last and x != blank_id:
                plt.text(i, 1, str(labels[x] if x < len(labels) else "-"), ha="center", va="bottom")
            last = x

    plt.show()
