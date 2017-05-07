import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions

import sys
import math

class MyDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        return self.dataset[i]

#単語のリストのリストをIDのリストのリストに変換する
#numpyの型に収めるため、-1をpaddingする(後で除外する)
def make_dataset(word2ind_dic, utt_lines, res_lines):
    vocab_size = len(word2ind_dic)

    utt_word_max = max([len(ws) for ws in utt_lines])
    res_word_max = max([len(ws) for ws in res_lines])
    max_word_num = max(utt_word_max, res_word_max)

    data_size = len(res_lines)

    dataset = []

    for utt_line, res_line in zip(utt_lines, res_lines):
        utt_ids = []
        for utt_word in utt_line:
            ind = word2ind_dic[utt_word]
            utt_ids.append(ind)
        utt_ids.append(2) #EOS

        for i in range(len(utt_line), max_word_num):
            utt_ids.append(-1)

        utt_ids_np = np.array(utt_ids, dtype=np.int32)

        res_ids = []
        for res_word in res_line:
            ind = word2ind_dic[res_word]
            res_ids.append(ind)
        res_ids.append(2) #EOS

        for i in range(len(res_line), max_word_num):
            res_ids.append(-1)

        res_ids_np = np.array(res_ids, dtype=np.int32)

        #タプルではないが…
        tpl = np.array([utt_ids_np, res_ids_np], dtype=np.int32)
        dataset.append(tpl)

    return np.array(dataset, dtype=np.int32)


class Encoder_chain(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder_chain, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, hidden_size * 4),
            hh = L.Linear(hidden_size, hidden_size * 4)
        )

    def __call__(self, x, c, h):
        #xはIDのバッチ
        e = F.tanh(self.xe(x))
        return F.lstm(c, self.eh(e)+ self.hh(h))


class Decoder_chain(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder_chain, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, hidden_size * 4),
            hh = L.Linear(hidden_size, hidden_size * 4),
            he = L.Linear(hidden_size, embed_size),
            ey = L.Linear(embed_size, vocab_size)
            )

    def __call__(self, y_id_batch, c, h):
        e = F.tanh(self.ye(y_id_batch))
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        t = self.ey(F.tanh(self.he(h)))
        return (t, c, h)

class Seq2seq_chain(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, word2ind_dic):
        super(Seq2seq_chain, self).__init__(
            encoder = Encoder_chain(vocab_size, embed_size, hidden_size),
            decoder = Decoder_chain(vocab_size, embed_size, hidden_size)
            )

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word2ind_dic = word2ind_dic
        self.ind2word_dic = {v:k for k, v in word2ind_dic.items()}

        self.c = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))
        self.h = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))

    #1文に含まれる[単語のID]をencode部に入力し、
    #decode部に渡すための中間層を計算
    def encode(self, word_ids):
        c = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))
        h = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))

        for word_id in word_ids.data:
            #行列の形を合わせるために入れた負のIDは無視
            if(word_id < 0):
                continue

            word_id_batch = Variable(np.array([word_id], dtype=np.int32))
            c, h = self.encoder(word_id_batch, c, h)

        self.h = h
        self.c = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))

    def decode(self, res_id):
        res_id_batch = Variable(np.array([res_id], dtype=np.int32))
        t, c, h = self.decoder(res_id_batch, self.c, self.h)
        self.c = c
        self.h = h
        return t

    def reset(self):
        self.c = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))
        self.h = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))

        return


    def get_response(self, words):
        ids = [self.word2ind_dic[word] for word in words if word in self.word2ind_dic]
        np_ids = np.array(ids, dtype=np.int32)
        # sys.stderr.write(str(words) + "\n")
        # sys.stderr.write(str(np_ids) + "\n")
        utt_ids = Variable(np_ids)
        self.encode(utt_ids)

        t = 2 #</s>のidは2
        output_words = []
        while not (len(output_words) > 0 and (t == 2)):
            y = self.decode(t)
            y_ind = int(F.argmax(y).data)
            y_word = self.ind2word_dic[y_ind]

            output_words.append(y_word)
            t = y_ind

            # sys.stdout.write(y_word)
            # sys.stdout.write(" ")
            # sys.stdout.flush()

            # if len(output_words) >= 10:
            #     print()
            #     break
        return output_words


    def __call__(self, dataset):
        loss = Variable(np.zeros((), dtype=np.float32))
        word_cnt = 0

        for data in dataset:
            self.reset()
            utt_ids = data[0]
            res_ids = data[1]

            #エンコード
            self.encode(utt_ids)

            #最初の入力 : </s>
            t = 2 #</s>のidは2

            for res_id in res_ids.data:
                y = self.decode(t)
                t = res_id
                t_var = Variable(np.array([res_id], dtype=np.int32)) #int32に注意

                loss += F.softmax_cross_entropy(y, t_var)
                word_cnt += 1

        loss_per_word = 1.0 * loss / word_cnt
        p = math.exp(- loss_per_word.data)
        sys.stderr.write("%.2f %.2f\n" % (loss_per_word.data, p))
        return loss_per_word

def get_word2ind_dic(dic_path):
    word2ind_dic = {}
    with open(dic_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            lst = line.split(' ')

            ind = int(lst[0])
            word = lst[1]
            word2ind_dic[word] = ind
    return word2ind_dic

def main():
    #単語から語彙IDを返す辞書
    #vocab.txtには"<id> 単語"という形式で単語が保存されている
    #<id>は0から始まる
    #id=0は<unk>用

    word2ind_dic = get_word2ind_dic("vocab.txt")

    #data_dir/utt/wakati/*.wakati と data_dir/res/wakati/*.wakati からデータを読み込み、IDベクトルに変換する
    utt_lines = [] #単語のリストのリスト
    with open("utt.txt", 'r') as f:
        for line in f:
            line = line.rstrip()
            words = line.split(' ')
            utt_lines.append(words)

    res_lines = [] #単語のリストのリスト
    with open("res.txt", 'r') as f:
        for line in f:
            line = line.rstrip()
            words = line.split(' ')
            res_lines.append(words)

    utt_lines = utt_lines[:100]
    res_lines = res_lines[:100]
    vocab_size = len(word2ind_dic)
    embed_size  = 300
    hidden_size = 300
    model = Seq2seq_chain(vocab_size, embed_size, hidden_size, word2ind_dic)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    dataset = make_dataset(word2ind_dic, utt_lines, res_lines)
    train = MyDataset(dataset)

    train_iter= chainer.iterators.SerialIterator(train, batch_size=50, repeat=True, shuffle=True)
    updater = training.StandardUpdater(train_iter, optimizer)

    trainer = training.Trainer(updater, (80, 'epoch'))

    #LogReportとPrintReportのkeysの設定がよく分かっていない。
    # keys=['epoch', 'main/loss'],
    trainer.extend(extensions.LogReport(log_name="log", trigger=(1, 'epoch')))
    # trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()
    serializers.save_npz('seq2seq.model', model)

    return 0

if __name__ == '__main__':
    sys.exit(main())
