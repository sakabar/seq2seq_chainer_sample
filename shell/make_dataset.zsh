#!/bin/zsh

set -u

corpus_dir=/Users/sak/tmpDownloads/corpus #今回は対話破綻コーパス (https://sites.google.com/site/dialoguebreakdowndetection/chat-dialogue-corpus) を使用した。コーパスをダウンロードして解凍し、そこへのパスをここに設定する。
output_dir=data_dir
mkdir -p $output_dir/utt/txt
mkdir -p $output_dir/res/txt

mkdir -p $output_dir/utt/wakati
mkdir -p $output_dir/res/wakati


unset PYTHONPATH

output=`mktemp -t make_dataset`
for f in $corpus_dir/json/*/*.json; do
    python2.7 $corpus_dir/show_dial.py $f | grep "^[US]" > $output
    len=$(cat $output | wc -l )
    half_len=$[ $len / 2]
    use_len=$[ $half_len * 2 ]

    head -n $use_len $output | gawk 'NR % 2 == 1 {print $0}' | gsed -e 's/^S://' | gawk '{print $1}' | nkf -w -Z1 | gsed -e 's/ \+//g' | tee $output_dir/utt/txt/$f:t:r:r".txt" | mecab -O wakati > $output_dir/utt/wakati/$f:t:r:r".wakati"
    head -n $use_len $output | gawk 'NR % 2 == 0 {print $0}' | gsed -e 's/^U://' | gawk '{print $1}' | nkf -w -Z1 | gsed -e 's/ \+//g' | tee $output_dir/res/txt/$f:t:r:r".txt"| mecab -O wakati > $output_dir/res/wakati/$f:t:r:r".wakati"
done

rm -rf $output
