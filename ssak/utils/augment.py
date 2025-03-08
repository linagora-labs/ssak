import os
import random

import numpy as np
import soxbindings as sox

# Not mandatory?
import torch
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    BandStopFilter,  # FrequencyMask,
    ClippingDistortion,
    Gain,
    PitchShift,
    TimeStretch,
)

from .augment_reverberation import Reverberation

_class2name = {
    AddGaussianNoise: "AddGaussianNoise",
    AddBackgroundNoise: "AddBackgroundNoise",
    ClippingDistortion: "ClippingDistortion",
    BandStopFilter: "BandStopFilter",
    Gain: "Gain",
    TimeStretch: "TimeStretch",
    PitchShift: "PitchShift",
    Reverberation: "Reverberation",
}


def parameter2str(val):
    if isinstance(val, str):
        # Remove path and extension from filenames
        return val.split("/")[-1].split(".")[0]
    if isinstance(val, float):
        # String with 3 significant digits
        return f"{val:.3g}"
    return str(val)


def transform2str(transform, short=False):
    if transform == None:
        return ""
    s = _class2name[type(transform)]
    d = {}
    for k, v in sorted(transform.parameters.items()):
        if k in ["should_apply", "noise_start_index", "noise_end_index"]:
            continue
        if short:
            d.update({"".join([a[0] for a in k.replace("-", "_").split("_")]): parameter2str(v)})
        else:
            d.update({k: str(v).split("/")[-1]})
    if len(d) == 0:
        return s
    if short and len(d) == 1:
        return s + "_" + list(d.values())[0]
    if short:
        return s + "_" + "_".join([k + ":" + v for k, v in d.items()])
    return s + " (" + ", ".join([k + ":" + v for k, v in d.items()]) + ")"


class SpeechAugment:
    def __init__(
        self,
        sample_rate=16000,
        apply_prob=0.5,
        speed=True,
        gain=True,
        pitch=False,
        distortion=False,
        bandstop=False,
        gaussian_noise=False,
        noise_dir=None,  # "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises"
        rir_dir=None,  # "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise"
        rir_lists=None,  # ["simulated_rirs_16k/smallroom/rir_list", "simulated_rirs_16k/mediumroom/rir_list", "simulated_rirs_16k/largeroom/rir_list"]
        verbose=False,
        save_audio_dir=None,
        max_saved_audio=100,
        apply_speed_separately=True,
    ):
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.transforms = []
        if gaussian_noise:
            self.transforms += [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0)]
        if distortion:
            self.transforms += [ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=30, p=1.0)]
        if bandstop:
            self.transforms += [BandStopFilter(p=1.0)]  # FrequencyMask(min_frequency_band=0.2, max_frequency_band=0.4, p=1.0)
        if gain:
            self.transforms += [Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0)]
        if pitch:
            self.transforms += [PitchShift(min_semitones=-2, max_semitones=2, p=1.0)]
        if noise_dir is not None:
            self.transforms += [AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=5, max_snr_in_db=50, p=1.0)]
        if rir_dir is not None:
            assert rir_lists, "rir_lists must be provided iff rir_dir is provided"
            self.transforms += [Reverberation(path_dir=rir_dir, rir_list_files=rir_lists, sample_rate=sample_rate, p=1.0)]
        else:
            assert not rir_dir, "rir_dir must be provided iff rir_lists is provided"
        # Speed at the end
        if speed:
            self.transforms += [TimeStretch(min_rate=0.95, max_rate=1.05, leave_length_unchanged=False, p=1.0)]

        self.num_trans = len(self.transforms)
        self.apply_speed_separately = apply_speed_separately and speed
        self.verbose = verbose
        self.save_audio_dir = save_audio_dir
        if save_audio_dir:
            os.makedirs(save_audio_dir, exist_ok=True)
        self.save_audio_idx = 0
        self.num_saved_audio = 0
        self.max_saved_audio = max_saved_audio

    def save_audio(self):
        if not self.save_audio_dir:
            return False
        self.num_saved_audio += 1
        return self.num_saved_audio <= self.max_saved_audio

    def call_single(self, input_values):
        """apply a random data augmentation technique from a list of transformations"""

        coin = random.random() < self.apply_prob

        do_save_audio = self.save_audio()

        if coin or self.apply_speed_separately:
            is_torch = isinstance(input_values, torch.Tensor)
            if is_torch:
                input_values = input_values.numpy()
            elif isinstance(input_values, list):
                input_values = np.array(input_values)

            if coin:
                # Note: we could use some weights / probabilities here. See random.choices([1,2,3], weights=[0.2, 0.2, 0.7], k=10)
                if self.apply_speed_separately:
                    i_transform = random.randint(0, self.num_trans - 2)
                else:
                    i_transform = random.randint(0, self.num_trans - 1)
                transform = self.transforms[i_transform]
            else:
                transform = None

            if do_save_audio:
                self.save_audio_idx += 1
                os.makedirs(self.save_audio_dir, exist_ok=True)
                sox.write(os.path.join(self.save_audio_dir, f"{self.save_audio_idx:04d}.wav"), input_values, self.sample_rate)

            if transform is not None:
                input_values = np.array(transform(samples=input_values, sample_rate=self.sample_rate))
                if self.verbose:
                    print(transform2str(transform))

            if self.apply_speed_separately:
                input_values = self.transforms[-1](samples=input_values, sample_rate=self.sample_rate)
                if self.verbose:
                    print(transform2str(self.transforms[-1]))

            if do_save_audio:
                trstr = transform2str(transform, short=True) if transform is not None else ""
                if self.apply_speed_separately:
                    trstr += "_" + transform2str(self.transforms[-1], short=True)
                sox.write(
                    os.path.join(self.save_audio_dir, f"{self.save_audio_idx:04d}_{trstr}.wav"),
                    input_values,
                    self.sample_rate,
                )

            if is_torch:
                input_values = torch.tensor(input_values)

        return input_values

    def __call__(self, input_values, wav_len=None):  # TODO: use len information when it is a batch
        if self.is_batch(input_values):
            out = [self.call_single(input_values[0])]
            self.transforms[-1].freeze_parameters()
            out += [self.call_single(x) for x in input_values[1:]]
            self.transforms[-1].unfreeze_parameters()
            if isinstance(input_values, torch.Tensor):
                out = torch.stack(out)
            elif isinstance(input_values, np.ndarray):
                out = np.array(out)
            return out
        else:
            return self.call_single(input_values)

    def is_batch(self, input_values):
        if isinstance(input_values, list) and not isinstance(input_values[0], (int, float)):
            return True
        if hasattr(input_values, "shape") and len(input_values.shape) > 1:
            return True
        return False
