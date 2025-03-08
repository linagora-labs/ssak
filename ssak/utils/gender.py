from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from transformers.file_utils import ModelOutput
from transformers.models.hubert.modeling_hubert import HubertModel, HubertPreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


_gender_classifiers = {}
_gender_classifiers_device = {}


def predict_gender(
    waveform,
    sample_rate=16_000,
    device="cpu",
    model="m3hrdadfi/hubert-base-persian-speech-gender-recognition",
    output_type="best",
):
    """
    Predict gender of an audio waveform (with one speaker).

    Args:
        waveform (np.array): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        device (str): The device to run the model on (cpu or cuda).
        model (str): The model to use.
        output_type (str): The output type (best or scores).
            - best : The best prediction (either "m" or "f").
            - scores : The scores of both predictions in a dictionary (ex: {"m": 0.9, "f": 0.1}).
    """
    global _gender_classifiers, _gender_classifiers_device
    if model not in _gender_classifiers:
        config = AutoConfig.from_pretrained(model)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
        net_model = HubertForSpeechClassification.from_pretrained(model)
        net_model = net_model.eval().to(device)
        labels = dict((i, l.lower()) for (i, l) in config.id2label.items())
        assert "f" in labels.values(), f"No female label found in {model} (among {labels})"
        assert "m" in labels.values(), f"No male label found in {model} (among {labels})"
        assert len(labels) == 2, f"Expected 2 labels, found {len(labels)} in {model} ({labels})"
        _gender_classifiers[model] = (net_model, feature_extractor, labels)
        _gender_classifiers_device[model] = device
    else:
        (net_model, feature_extractor, labels) = _gender_classifiers[model]
        assert model in _gender_classifiers_device
        if device != _gender_classifiers_device[model]:
            net_model = net_model.to(device)
            _gender_classifiers_device[predict_gender] = device

    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    if device != "cpu":
        inputs = inputs.to(device)

    with torch.no_grad():
        logits = net_model(**inputs).logits

    scores = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    if output_type == "scores":
        return dict(zip(labels.values(), scores.tolist()))

    elif output_type == "best":
        # Get the index of the label with the highest score
        best_index = np.argmax(scores)

        # Get the label and score of the best prediction
        best_label = labels[best_index]
        return best_label

    else:
        raise ValueError(f"Unknown output type: {output_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gender prediction of an audio.")
    parser.add_argument("audio", type=str, help="Path to audio.", nargs="+")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds.")
    parser.add_argument("--model", type=str, default="m3hrdadfi/hubert-base-persian-speech-gender-recognition", help="Model.")
    args = parser.parse_args()

    import json

    from ssak.utils.audio import load_audio

    for audio in args.audio:
        audio = load_audio(audio, start=args.start, end=args.end)
        prediction = predict_gender(audio, output_type="scores")
        print(json.dumps(prediction, indent=2))
