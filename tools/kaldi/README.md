The main generic home scripts are:
- `kaldi_split.py`       : split dataset
- `kaldi_subsample.py`   : take a subset of the dataset
- `kaldi_stats.py`       : print statistics about the dataset (mostly about durations)

Some scripts to convert usual format to Kaldi:
- `tsv2kaldi.py`         : convert TSV format like CommonVoice
- `transcriber2kaldi.py` : convert Transcriber format

The main scripts taken from Kaldi are:
- `utils/get_utt2dur.sh`       : generate utt2dur file in a kaldi folder
- `utils/fix_data_dir.sh`      : fix kaldi folder
- `utils/validate_data_dir.sh` : validate kaldi folder
