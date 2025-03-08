set -e

if [ $# -ne 3 ];then
    echo "Usage: $0 <dir> <from> <to>"
    echo "Replace <from> by <to> in all wav.scp files found in <dir> and its subfolders"
    exit 1
fi

DIR=$1
FROM=`echo $2 | awk -F ":" '{gsub("/", "\\\\/"); print $NF}'`
TO=`echo $3 | awk -F ":" '{gsub("/", "\\\\/"); print $NF}'`

if [ ! -d $DIR ];then
    echo "Error: $DIR is not a directory"
    exit 1
fi

FILES=`find $DIR -type f -name wav.scp`

for FILE in $FILES;do
    if [ `grep $FROM $FILE | wc -l` -eq 0 ];then continue; fi
    echo "Modifying: $FILE"
    sed -i 's/'$FROM'/'$TO'/g' $FILE
done
