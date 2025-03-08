# Speech Swiss Army Knife (SSAK)

![SSAK logo](assets/logos/logo-SSAK.png)

This repository contains helpers and tools to train end-to-end ASR, and do inference with ASR.

It is based on SpeechBrain and HuggingFace's Transformers packages, which are both based on PyTorch.
It also includes inference with Vosk for (baseline) kaldi models.

The main data format is the one of Kaldi, i.e. folders with files:
```
├── [segments]   : utterance -> file id, start, end
├── text         : utterance -> annotation
├── utt2dur      : utterance -> duration (use tools/kaldi/utils/get_utt2dur.sh if you are missing this file)
└── wav.scp      : file id (or utterance if no segments) -> audio file [with sox/flac conversion]
```
and also optionally (not exploited in most cases):
```
├── spk2gender   : speaker -> gender
├── spk2utt      : speaker -> list of utterances
└── utt2spk      : utterance -> speaker
```

This repository focus on the following features:
- Text cleaning and normalization, to train and evaluate acoustic and language models
- Tools to manage labeled audio. For instance cut transcriptions into smaller chunks of audio, with corresponding timestamps
- Scripts to convert data into different formats
- Scripts to train models with common frameworks
- Scripts to decode with models from common frameworks

## Repository Folder Structure

```
├── ssak/      : Main python library
│   ├── infer/          : Functions and scripts to run inference and evaluate models
│   ├── train/          : Scripts to train models (transformers, speechbrain, ...)
│   └── utils/          : Helpers
├── tools/           : Scripts to cope with audio data (data curation, ...)
│   ├── kaldi/utils/    : Scripts to check and complete kaldi's data folders (.sh and .pl scripts)
│   ├── LeVoiceLab/     : Scripts to convert data from/to LeVoiceLab format (see https://speech-data-hub.levoicelab.org/)
│   └── scraping/       : Scripts to scrape a collection of documents (docx, pdf...) or the web
├── docker/          : Docker environment
└── tests/           : Unittest suite
    ├── data/           : Data to run the tests
    ├── expected/       : Expected outputs for some non-regression tests
    ├── unittests/      : Code of the tests
    └── run_tests.py    : Entrypoint to run tests
```

## Installation

### Requirements

```
sudo apt-get install \
        sox \
        libsox-fmt-mp3 \
        libsox-dev \
        ffmpeg \
        libssl-dev \
        libsndfile1 \
        python3-dev \
        portaudio19-dev \
        libcurl4-openssl-dev \
        xvfb

pip3 install -r requirements.txt
pip3 install -r tools/requirements.txt
```

For scraping tools you may also need additional dependencies:
```
sudo add-apt-repository ppa:mozillateam/ppa
sudo apt-get update
sudo apt-get install -y --no-install-recommends firefox-esr
```


### Docker

If not done, pull the docker image:
```
docker pull lintoai/sak:latest
```
or build it:
```
docker build -t lintoai/sak:latest .
```

Run it, with advised options:
```
docker run -it --rm \
    --shm-size=4g \
    --user $(id -u):$(id -g) \
    --env HOME=~ --workdir ~ \
    -v /home:/home \
    --name sak_workspace \
    lintoai/sak:latest
```
(also add `--gpus all` to use GPU).
