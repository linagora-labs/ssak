#!/bin/bash

set -e

__FILE__=`realpath $0`
FOLDER=`dirname $__FILE__`
FOLDER=`dirname $FOLDER`

echo $FOLDER
cd $FOLDER

MAINDIR=/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr

while [ 1 -gt 0 ];do

    python3 scrape_youtube_validate.py $MAINDIR -v --model /home/jlouradour/projects/SpeechBrain/best_model_speechbrain || exit 1
    for folder in \
        "$MAINDIR/mp3_checked_fr_noasr_nostream_stt" \
        "$MAINDIR/mp3_checked_fr_noasr_stream_stt" \
        ;do
        python3 ../total_duration.py $folder
    done
    sleep 3600

done