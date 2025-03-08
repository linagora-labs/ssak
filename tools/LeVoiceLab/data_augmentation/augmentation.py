import random

import numpy as np
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    BandStopFilter,  # FrequencyMask,
    ClippingDistortion,
    Gain,
    PitchShift,
    TimeStretch,
)
from augmentations.reverberation import Reverberation

_class2name = {
    AddGaussianNoise: "GaussianNoise",
    AddBackgroundNoise: "BackgroundNoise",
    ClippingDistortion: "ClippingDistortion",
    BandStopFilter: "BandStopFilter",
    PitchShift: "PitchShift",
    Reverberation: "Reverberation",
    Gain: "Gain",
    TimeStretch: "TimeStretch",
}

_class2descr = {
    AddGaussianNoise: "Ajout de bruit Gaussien",
    AddBackgroundNoise: "Ajout de bruit de fond",
    ClippingDistortion: "distorsion par clipping",
    BandStopFilter: "Filtre passe bande",
    PitchShift: "Modification du pitch de la voix",
    Reverberation: "Simulation de réverbération",
    Gain: "Modification du gain",
    TimeStretch: "Extension du temps",
}


def parameter2str(val):
    if isinstance(val, str):
        # Remove path and extension from filenames
        return val.split("/")[-1].split(".")[0]
    if isinstance(val, float):
        # String with 3 significant digits
        return f"{val:.3g}"
    return str(val)


def transform2genericstr(transform):
    if transform == None:
        return ""
    return _class2name[type(transform)]


def transform2description(transform):
    return _class2descr[type(transform)]


def transform2str(transform, short=False):
    if transform == None:
        return ""
    s = transform2genericstr(transform)
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
        noise_dir=None,  # "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises"
        rir_dir=None,  # "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise"
        rir_lists=None,  # ["simulated_rirs_16k/smallroom/rir_list", "simulated_rirs_16k/mediumroom/rir_list", "simulated_rirs_16k/largeroom/rir_list"]
        apply_prob=0.5,
    ):
        self.apply_prob = apply_prob
        self.transforms = [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),
            ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=30, p=1.0),
            BandStopFilter(p=1.0),  # FrequencyMask(min_frequency_band=0.2, max_frequency_band=0.4, p=1.0),
            # Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0), # Not very interesting
            # TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=False, p=1.0), # Not realistic + harder to handle (needs to change all times)
            PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
            # Trim(p=1.0),
        ]
        if noise_dir is not None:
            self.transforms += [AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=5, max_snr_in_db=50, p=1.0)]

        if rir_dir is not None and rir_lists is not None:
            self.transforms += [Reverberation(path_dir=rir_dir, rir_list_files=rir_lists, p=1.0)]

        self.num_trans = len(self.transforms)
        self.i_trans = -1

    def get_num_transforms(self):
        assert self.num_trans == len(self.transforms)
        return len(self.transforms)

    def __call__(self, input_values, sample_rate):
        """apply a random data augmentation technique from a list of transformations"""
        transform = None
        if random.random() < self.apply_prob:
            # TODO: use
            # random.choices([1,2,3], weights=[0.2, 0.2, 0.7], k=10)
            # i_trans = random.randint(0, self.num_trans - 1)
            self.i_trans = self.i_trans + 1
            i_trans = self.i_trans % self.num_trans
            transform = self.transforms[i_trans]
            input_values = np.array(transform(samples=np.array(input_values), sample_rate=sample_rate))
        return transform, input_values
