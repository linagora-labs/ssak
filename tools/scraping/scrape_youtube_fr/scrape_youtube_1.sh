#!/bin/bash

set -e

__FILE__=`realpath $0`
FOLDER=`dirname $__FILE__`
FOLDER=`dirname $FOLDER`

cd $FOLDER

python3 scrape_youtube.py \
    /media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr \
    --search_query /media/nas/CORPUS_PENDING/Corpus_text/FR/SCRAPED_NEWS/all.txt \
    --ngram 3,2,1

