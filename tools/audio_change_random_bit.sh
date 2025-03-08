#!/bin/bash
set -uea

if [ $# -ne 2 ];then
    echo "Usage: $0 <input> <output>"
    echo "Change a random bit in the input file and produce an output file"
    exit 1
fi

input=$1
output=$2

if [ ! -f $input ];then
    echo "Error: $input does not exist"
    exit 1
fi
if [ "$input" == "$output" ];then
    echo "Error: $input and $output are the same file"
    exit 1
fi

input_extension=`echo $input | rev | cut -d. -f1 | rev | tr '[:upper:]' '[:lower:]'`
output_extension=`echo $output | rev | cut -d. -f1 | rev | tr '[:upper:]' '[:lower:]'`

# Convert mpga and mp3 files
tmpfile=""
if [ "$input_extension" == "mpga" ];then
    tmpfile=`mktemp --suffix=.mp3`
    # ffmpeg -i $input -y -acodec pcm_s16le -ac 1 -ar 16000 $tmpfile
    # ffmpeg -i $input -y -vn -ar 44100 -ac 2 -b:a 192k $tmpfile
    ffmpeg -i $input -y -vn $tmpfile
    input=$tmpfile
fi
if [ "$output_extension" == "mpga" ];then
    output_mp3=`mktemp --suffix=.mp3`
    sox $input $output_mp3 gain -n 1
    ffmpeg -i $output_mp3 -acodec mp2 $output
    rm $output_mp3
else
    # Main command
    sox $input $output gain -n 1
fi

if [ -n "$tmpfile" ];then
    rm $tmpfile
fi