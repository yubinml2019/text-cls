import numpy as np
from gensim.models import KeyedVectors

# 对tx词向量过滤
"""
过滤标点符号
过滤停用词
过滤单个中文
过滤不关心的词性（如副词）
过滤非中文词（马云和、马云说、如马云，只是思路，答主目前没想到好的实现办法）

"""
from tqdm import tqdm
import numpy as np
import gensim


def load_embedding(path):
    """

    :param path:
    :return:
    """
    embedding_index = {}
    f = open(path, encoding='utf8')
    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    return embedding_index


# embedding_index = load_embedding("./Tencent_AILab_ChineseEmbedding.txt")
with open('./clean_Tencent_AILab_ChineseEmbedding.txt', 'a') as w_f:
    with open('./Tencent_AILab_ChineseEmbedding.txt', 'r', errors='ignore')as f:
        for i in tqdm(range(8825659)):
            data = f.readline()
            a = data.split()
            if i == 0:
                w_f.write('8748464 200\n')
            if len(a) == 201:
                if a[0].isdigit():
                    continue
                else:
                    w_f.write(data)

model = KeyedVectors.load_word2vec_format('./clean_Tencent_AILab_ChineseEmbedding.txt', binary=False,
                                          unicode_errors='ignore')
model.save("./clean_tx_embedding")
model.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar


def compute_ngrams(word, min_n, max_n):
    # BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return list(set(ngrams))


def wordVec(word, wv_from_text, min_n=1, max_n=3):
    '''
    ngrams_single/ngrams_more,主要是为了当出现oov的情况下,最好先不考虑单字词向量
    '''
    # 确认词向量维度
    word_size = wv_from_text.wv.syn0[0].shape[0]
    # 计算word的ngrams词组
    ngrams = compute_ngrams(word, min_n=min_n, max_n=max_n)
    # 如果在词典之中，直接返回词向量
    if word in wv_from_text.wv.vocab.keys():
        return wv_from_text[word]
    else:
        # 不在词典的情况下
        word_vec = np.zeros(word_size, dtype=np.float32)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams if len(ng) > 1]
        # 先只接受2个单词长度以上的词向量
        for ngram in ngrams_more:
            if ngram in wv_from_text.wv.vocab.keys():
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
                # print(ngram)
        # 如果，没有匹配到，那么最后是考虑单个词向量
        if ngrams_found == 0:
            for ngram in ngrams_single:
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
        if word_vec.any():
            return word_vec / max(1, ngrams_found)
        else:
            raise KeyError('all ngrams for word %s absent from model' % word)


vec = wordVec('VIP', model, min_n=2, max_n=3)  # 词向量获取
model.most_similar(positive=[vec], topn=10)  # 相似词查找
oov_path = "./all/mtext_cls_tf1/data/balance_data_jieba/oov_vocab.txt"
oov_vocab = []
with open(oov_path) as f:
    oov_vocab = [word.strip() for word in f.readlines()]
word_sims = {}
for oov_w in oov_vocab[:10]:
    vec = wordVec(oov_w, model, min_n=1, max_n=len(oov_w))  # 词向量获取

    print(model.most_similar(positive=[vec], topn=10))  # 相似词查找