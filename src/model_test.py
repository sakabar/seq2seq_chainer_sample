import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions

import sys


import model_train
from model_train import Seq2seq_chain

def main():
    word2ind_dic = model_train.get_word2ind_dic("vocab.txt")
    vocab_size  = len(word2ind_dic)
    embed_size  = 300
    hidden_size = 300
    model = Seq2seq_chain(vocab_size, embed_size, hidden_size, word2ind_dic)
    serializers.load_npz('seq2seq.model', model)

    print("Ready.")
    for line in sys.stdin:
        line = line.rstrip()
        words = line.split(' ')
        output_words = model.get_response(words)
        print(" ".join(output_words))

    return 0

if __name__ == '__main__':
    sys.exit(main())
