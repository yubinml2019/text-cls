import jieba
import os
import pandas as pd
import glob
import random
import tensorflow.keras as kr
import numpy as np
import pkuseg
from collections import Counter


def get_stopwords(file_path):
    with open(file_path) as f:
        return [word.strip() for word in f.read()]


def read_file(file_path):
    contents, labels = [], []
    with open(file_path) as f:
        for line in f.readlines():
            try:
                content, label = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels


def read_file_with_sep(file_path, sep):
    contents, labels = [], []
    with open(file_path) as f:
        for line in f.readlines():
            try:
                content, label = line.strip().split(sep)
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels


def cut_files(in_dir, out_dir, dict_path = None):
    """

    :param in_dir: 输入文件路径 train test dev 未分词
    :param out_dir: 输出路径 分词之后的
    :param stopwords: 停用词的list
    :return:
    """

    stopwords = get_stopwords("./data/stopwords.txt")
    if dict_path:
        jieba.load_userdict(dict_path)
    for file in sorted(glob.glob(in_dir)):
        sentences = []
        with open(file) as f:
            for line in f.readlines():
                ss = line.strip().split("\001")
                label = ss[0]
                content = ss[1]
                label = ss[2]
                try:
                    segs = jieba.cut(line.strip() + content, cut_all=True)
                    segs = list(filter(lambda x: len(x) > 1, segs))
                    segs = list(filter(lambda x: x not in stopwords, segs))
                    sentences.append(" ".join(segs) + "\t" + label)
                except Exception as e:
                    print(line)

        type = file.split("/")[-1].split(".")[0]
        print(type)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        w_file = os.path.join(out_dir, type + "_cuts.txt")
        print(w_file)
        with open(w_file, "w") as wf:
            wf.write("\n".join(sentences))


def build_vocab(data_path, vocab_path, vocab_size=40000):
    """

    :param data_dir: 训练数据目录
    :param vocab_path: 输出词表地址
    :param vocab_size: 词表大小
    :return:
    """
    train_data, label = read_file(data_path)
    train_data = [word for line in train_data for word in line.split()]
    counter = Counter(train_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = zip(*count_pairs)
    words = ["<UNK>"] + list(words)
    with open(vocab_path, "w") as wf:
        wf.write("\n".join(words))

def word_freq(file_path):
    contents, labels = read_file(file_path)
    words = {}
    for line in contents:
        for w in line:
            if w not in words:
                words[w] = 1
            else:
                words[w] = words[w] + 1
    return words


def read_category(data_path):
    _, cate = read_file(data_path)
    cate = set(cate)
    cate2id = dict(zip(cate, range(len(cate))))
    return cate2id


def read_vocab(vocab_path):
    words = []
    with open(vocab_path) as f:
        [words.append(word.strip()) for word in f.readlines()]
    word2id = dict(zip(words, range(len(words))))
    return word2id

def save_embedding(embedding_dict, save_path):

    with open(save_path, "w") as  f:
        f.write("\n".join([k + "\t" + " ".join(map(str, v.tolist())) for k, v in embedding_dict.items()]))


def build_vocab_from_list(embedding_keys, file_path):
    with open(file_path, "w") as f:
        f.write("\n".join(embedding_keys))


def analysis_cut(cut_tool, cut_dir, embedding_index):
    """

    :param cut_tool:  分词工具 "jieba"
    :param cut_dir:  "分词后的文件存储目录
    :param embedding_index: 预训练的腾讯词向量
    :return:
    """
    contents, labels = read_file(os.path.join(cut_dir, "train_cuts.txt"))
    words = {}
    for line in contents:
        for w in line.strip().split():
            if w in words:
                words[w] = words[w] + 1
            else:
                words[w] = 1


    sorted_words = sorted(words.items(), key = lambda x:x[1], reverse=True)
    filtered2 = {k:v  for k, v in words.items() if v >=2 }  # 15361 词频>=2
    filtered1 = {k:v  for k, v in words.items() if v ==1 }
    print("%s分词 词频大于1的个数 %d" %(cut_tool, len(filtered2)))
    if cut_tool == "jieba":
        cut_vocab = filtered2.keys()
    else:
        cut_vocab = words.keys()
    pre_embeddings = {}
    not_in = []
    for word in cut_vocab:
        if word in embedding_index:
            pre_embeddings[word] = embedding_index[word]
        else:
            not_in.append(word)

    print("%s分词在腾讯预训练词向量中有 %d"%(cut_tool, len(pre_embeddings)))
    print("%s分词在腾讯预训练词向量中有 占比 %.3f"%(cut_tool,len(pre_embeddings)*1.0 / len(words)))
    build_vocab_from_list(pre_embeddings.keys(), os.path.join(cut_dir,".pre_embedding_vocab.txt"))
    build_vocab_from_list(not_in, os.path.join(cut_dir, "oov_vocab.txt"))
    save_embedding(pre_embeddings, os.path.join(cut_dir,"pre_embedding.txt"))
# analysis_cut("jieba", "./balance_data_jieba/")
# analysis_cut("pkuseg", "./balance_data_pkuseg/")

def encode_file(file_path, word2id, cate2id, max_seqlen):
    contents, labels = read_file(file_path)
    content_ids, label_ids = [], []
    for line in contents:
        sentence = []
        for word in line.split():
            if word2id.get(word) != None:
                sentence.append(word2id[word])
            else:
                sentence.append(word2id["<UNK>"])
        content_ids.append(sentence)
    label_ids = [cate2id[label] for label in labels]

    x_pad = kr.preprocessing.sequence.pad_sequences(content_ids, max_seqlen, padding="post")
    y_pad = kr.utils.to_categorical(label_ids, num_classes=len(cate2id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=128, shuffle=True):
    # num_batch = int((len(x) - 1 / batch_size)) + 1
    num_batch = int(len(x) / batch_size)
    data = list(zip(x, y))
    if shuffle:
        random.shuffle(data)
    x, y = zip(*data)

    for i in range(num_batch):
        start = i * batch_size
        end = min(start + batch_size, len(x))
        yield x[start: end], y[start: end]

def get_file_lines(file_path):
    with open(file_path) as f:
        return len(f.readlines())

def check_test_data(file_path, predicts, labels, id2cate):
    df = pd.read_csv(file_path, sep="\001", names=["content", "true_label"]).head(len(predicts))
    df1 = pd.DataFrame({"predict": [id2cate[i] for i in predicts],
                        "label": [id2cate[i] for i in labels]})
    df = pd.concat([df, df1], axis=1)
    df.to_csv("./test_result.txt", sep="\001", index=False, header=None)


def pkuseg_cut_files(in_dir, out_dir):
    """

    :param in_dir: 输入文件路径 train test dev 未分词
    :param out_dir: 输出路径 分词之后的
    :param stopwords: 停用词的list
    :return:
    """
    seg = pkuseg.pkuseg(model_name="web")

    stopwords = get_stopwords("./data/stopwords.txt")
    for file in sorted(glob.glob(in_dir)):
        sentences = []
        with open(file) as f:
            for line in f.readlines():
                ss = line.strip().split("\001")
                line = ss[0]
                content = ss[1]
                label = ss[2]
                try:
                    segs = seg.cut(line + content)
                    segs = list(filter(lambda x: len(x) > 1, segs))
                    segs = list(filter(lambda x: x not in stopwords, segs))
                    sentences.append(" ".join(segs) + "\t" + label)
                except Exception as e:
                    print(line)

        type = file.split("/")[-1].split(".")[0]
        print(type)
        w_file = os.path.join(out_dir, type + "_cuts.txt")
        print(w_file)
        with open(w_file, "w") as wf:
            wf.write("\n".join(sentences))

def analysis_max_seq(file_path, percent):

    """

    :param file_path: 分词后的train/val 文件路径
    :param percent: 想要查看多少百分比
    :return:
    """
    words_in_line_num = []
    with open(file_path) as f:
        for line in f.readlines():
            ss = line.split("\t")
            words_num = len(ss[0].split())
            words_in_line_num.append(words_num)

    return np.percentile(words_in_line_num, percent)


def load_embedding(path):
    """

    :param path:
    :return:
    """
    embedding_index = {}
    f = open(path,encoding='utf8')
    for index,line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()

    return embedding_index

#embedding_index = load_embedding('./Tencent_AILab_ChineseEmbedding.txt')

cut_files("./data/balance_data/*.txt", "./data/balance_data_jieba_allcut_userdict")
#pkuseg_cut_files("./data/balance_data/*.txt", "./data/balance_data_pkuseg")