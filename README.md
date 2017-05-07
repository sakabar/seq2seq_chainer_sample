# seq2seq_chainer_sample
Sample Implementation of Seq2Seq via Chainer

Seq2seqのサンプル実装です。

## 手順
* [雑談対話コーパス](https://sites.google.com/site/dialoguebreakdowndetection/chat-dialogue-corpus)をダウンロードして解凍します
* `shell/make_dataset.zsh`内の`corpus_dir`をダウンロードしたコーパスへのパスに書き換えます 
* `shell/make_dataset.zsh`でコーパス内の対話を分かち書きし、`data_dir/{utt,res}/wakati`内に格納します
 * 本来は破綻した対話は除外するべきですが、今回は動くものができあがれば良かったためそのまま使っています 
 * 自分で用意したファイルを`data_dir/utt/txt/hoge.txt`と`data_dir/res/txt/hoge.txt`に置く場合、それぞれの行数が一致している必要があります
* `make_utt_and_res.zsh`と`make_vocab_dic.zsh`を実行し、必要なファイルを生成します
* `python3.5 src/model_train.py`を実行すると、モデルの学習ができます
* 学習したモデルは、`echo "これは1つ目の文です。\nこれは2つ目の文です。" | mecab -O wakati | python3.5 src/model_test.py`などのように文を分かち書きして標準入力から渡すと返答が出力されます。


## 参考サイト
[今更ながらchainerでSeq2Seq(1)](http://qiita.com/kenchin110100/items/b34f5106d5a211f4c004#encoder)


## メモ: Chainerの初歩的引っかかりポイント
* 引数はバッチで渡すか、1つのベクトルか
    * 公式ドキュメントで確認しましょう
* 今自分が使おうとしている変数は`numpy.array`なのか`chainer.Variable`なのか
* `numpy.array`で多次元配列を作ろうとする場合、各次元の長さが同じになるようにpaddingを(手動で?)行う必要がある
 * 例: `[[0.1, 0.2], [0.3, 0.4, 0.5]]`を`numpy.array`に変換しようとするとエラーが出る。`[[0.1, 0.2, -1], [0.3, 0.4, 0.5]]`とでもしておいて、後で`-1`を消す