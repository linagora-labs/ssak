set -e

if [ $# -ne 1 ];then
    echo "Usage: $0 <dir>"
    echo "Check that all audio files mentioned in all wav.scp files found in <dir> (and its subfolders) exist"
    exit 1
fi

DIR=$1
QUICK=0
ECHO=1

find $DIR -name wav.scp | while read WAVSCP;do

    WAVSCPNAME=$WAVSCP

    if [ $QUICK -gt 0 ];then
        if [ $ECHO -gt 0 ];then
            echo "Checking last line of $WAVSCP"
        fi
        tail -n 1 $WAVSCP > /tmp/wav.scp
        WAVSCP=/tmp/wav.scp
    else
        if [ $ECHO -gt 0 ];then
            echo "Full check of $WAVSCP"
        fi
    fi

    if [ `cat $WAVSCP | wc -w` -eq 0 ];then
        echo "===================="
        echo "$WAVSCPNAME is empty"
        continue
    fi

    cat $WAVSCP | while read LINE;do
        I=2
        AUDIOFILE=`echo $LINE | awk '{print $'$I'}'`
        if [ "$AUDIOFILE" == "sox" ];then
            I=3
            AUDIOFILE=`echo $LINE | awk '{print $'$I'}'` 
        fi
        if [ "$AUDIOFILE" == "flac" ];then
            I="(NF-1)"
            AUDIOFILE=`echo $LINE | awk '{print $'$I'}'` 
        fi
        if [ `echo $AUDIOFILE | grep ^\' | wc -w` -gt 0 ];then
            AUDIOFILE=`echo $LINE | awk -F"'" '{print $2}'`
        fi
        if [ ! -f "$AUDIOFILE" ];then
            echo "===================="
            echo "Problem with $WAVSCPNAME"
            echo "$AUDIOFILE does not exist"
            break
        fi
    done
done
