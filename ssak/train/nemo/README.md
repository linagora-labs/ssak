# Fintuning Parakeet for french ASR

## Data

In NeMo, the files used for training are manifest files. Here is an example of row:
```
{"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt": "utterance_id"}
```

The mandatory fields are : audio_filepath, text and duration. Offset is needed when segments are only a part of the audio file. Others are optionals and don't seem to be use anywhere.

### Kaldi to NeMo

```
python tools/nemo/convert_kaldi_dataset_to_nemo.py INPUT_KALDI_DATASET_FOLDER OUTPUT_NEMO_FOLDER OUTPUT_AUDIO_PROCESSED_FOLDER
```

The script will convert the kaldi dataset to manifest file and will keep speaker informations. It will also transform the adui files if needed (to 16kHz and mono channel).

### HuggingFace to NeMo

For consistency, it is better to not use this.

#### Common Voice

```
python tools/nemo/convert_hf_dataset_to_nemo.py output_dir=datasets/nemo_dataset path="mozilla-foundation/common_voice_6_1" name="fr" split="validation" ensure_ascii=False use_auth_token=true
python tools/nemo/convert_hf_dataset_to_nemo.py output_dir=datasets/nemo_dataset path="mozilla-foundation/common_voice_6_1" name="fr" split="test" ensure_ascii=False use_auth_token=true
python tools/nemo/convert_hf_dataset_to_nemo.py output_dir=datasets/nemo_dataset path="mozilla-foundation/common_voice_6_1" name="fr" split="train" ensure_ascii=False use_auth_token=true
```

### Manifest files

You can clean manifest files from non french characters using the following command

```
python tools/nemo/clean_manifest_text_fr.py datasets/nemo_dataset/common_voice_6_1/fr/validation/validation_mozilla-foundation_common_voice_6_1_manifest.json datasets/nemo_dataset/common_voice_6_1/fr/validation/validation_mozilla-foundation_common_voice_6_1_manifest_clean.json
```

## Tokenizer

```
python tools/nemo/process_asr_text_tokenizer.py --manifest PATH_TO_MANIFEST_FILE --data_root tokenizer --vocab_size 1024 --tokenizer "spe" --spe_split_digits --log
python tools/nemo/process_asr_text_tokenizer.py --manifest datasets/nemo_dataset/common_voice_6_1/fr/validation/validation_mozilla-foundation_common_voice_6_1_manifest.json,datasets/nemo_dataset/common_voice_6_1/fr/test/test_mozilla-foundation_common_voice_6_1_manifest.json --data_root tokenizer --vocab_size 1024 --spe_split_digits --tokenizer "spe" --log
```

Create a tokenizer based on the manifest file
- vocab_size = 1024 for parakeet on HuggingFace



## Data processing pipeline

You can do everything said above in one command using:

```
bash tools/nemo/prepare_data.sh TOKENIZER_SIZE FILE_CONTAINING_PATH_TO_KALDI_DATASETS OUTPUT_PROCESSED_AUDIO_FOLDER OUTPUT_TARRED_DATASET
```

- TOKENIZER_SIZE : Size of the vocabulary of the tokenizer
- FILE_CONTAINING_PATH_TO_KALDI_DATASETS : A file containing the path to all the kaldi datasets you want to use (one line per dataset)
- OUTPUT_PROCESSED_AUDIO_FOLDER :  A folder to save the transformed audios 
/!\ Be aware that it requires a lot of disk space if you need to transform a lot of files
- OUTPUT_TARRED_DATASET : A folder to save the different buckets which will contain tar files 
/!\ Be aware that it requires a lot of disk space

In detail, the pipeline will convert kaldi datasets to manifest files, transform audios files if needed to the right format, merge the dataset, clean it, generate a tokenizer, split it to train/valid/test and finally makes buckets from the train split.

## TRAINING

```
python ssak/train/nemo/train.py --config-path FOLDER_CONTAINING_CONFIG --config-name CONFIG_FILE [overrides]
```

For example:
```
python ssak/train/nemo/train.py --config-path ssak/train/nemo/yamls  --config-name finetuning.yaml init_from_pretrained_model=stt_en_citrinet_512 model.freeze_encoder=False
```