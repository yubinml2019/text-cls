import jieba
import os
import glob
import random
import tensorflow.keras as kr


from collections import Counter
def get_stopwords(file_path):
    with open(file_path) as f:
        return [word.strip() for word in f.read()]

def read_file(file_path):
    contents, labels = [], []
    with open(file_path) as f :
        for line in f.readlines():
            try:
                content, label= line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels


def cut_files(in_dir, out_dir):
    """

    :param in_dir: 输入文件路径 train test dev 未分词
    :param out_dir: 输出路径 分词之后的
    :param stopwords: 停用词的list
    :return:
    """

    stopwords = get_stopwords("./data/stopwords.txt")
    for file in sorted(glob.glob(in_dir)):
        sentences = []
        with open(file) as f:
            for line in f.readlines():
                ss = line.strip().split("\t")
                line = ss[0]
                label = ss[1]
                try:
                    segs = jieba.lcut(line.strip())
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


def build_vocab(data_path, vocab_path, vocab_size = 40000):
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
    print(words)
    words = ["<UNK>"] + list(words)
    with open(vocab_path, "w") as wf:
        wf.write("\n".join(words))


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

def encode_file(file_path, word2id, cate2id):

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

    x_pad = kr.preprocessing.sequence.pad_sequences(content_ids, 10, padding="post")
    y_pad = kr.utils.to_categorical(label_ids, num_classes=len(cate2id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size = 64):
    num_batch = int((len(x) - 1 / batch_size)) + 1
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)

    for i in range(num_batch):
        start = i * batch_size
        end = min(start + batch_size, len(x))
        yield x[start: end], y[start: end]

