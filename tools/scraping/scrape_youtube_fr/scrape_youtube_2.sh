#!/bin/bash


set -e

__FILE__=`realpath $0`
FOLDER=`dirname $__FILE__`
FOLDER=`dirname $FOLDER`

echo $FOLDER
cd $FOLDER

MAINDIR=/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr

while [ 1 -gt 0 ];do

    echo "Download mp4..."
    python3 scrape_youtube_download_audio.py $MAINDIR || exit $?

    echo "Get metadata..."
    python3 scrape_youtube_get_metadata.py \
        $MAINDIR/fr \
        -o $MAINDIR/metadata \
        --ignore_if_exists \
        -c yt_scrape_check.tsv \
        -p yt_scrape_metadata.json || exit $?

    echo "Measure timing..."
    python3 ../total_duration.py $MAINDIR
    sleep 3600

done