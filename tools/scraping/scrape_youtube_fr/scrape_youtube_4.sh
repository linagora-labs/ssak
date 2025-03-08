#!/bin/bash
DIR="YouTubeFr"

rm -Rf $DIR/kaldi

for STREAM in "nostream" "stream";do

    python3 scrape_youtube_to_kaldi.py \
        $DIR"/mp3_checked_fr_noasr_"$STREAM"_stt" \
        $DIR"/fr_checked_fr_noasr_"$STREAM"_stt_formatted" \
        $DIR"/kaldi_"$STREAM
    
done