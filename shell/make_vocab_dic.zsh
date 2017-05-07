#!/bin/zsh
set -u

vocab_file=vocab.txt

{
  echo "<unk>"
  echo "<s>"
  echo "</s>"
  cat data_dir/{utt,res}/wakati/*.wakati | tr ' ' '\n' | grep -v "^$" | LC_ALL=C sort | uniq
} | awk '{id = NR-1; print id" "$0}' > $vocab_file

