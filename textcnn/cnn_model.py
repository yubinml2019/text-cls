import tensorflow as tf
from sklearn.metrics import confusion_matrix


class CNNConfig(object):
    max_seq_len = 500
    train_vocab_size = 1463
    #wc -l 统计文件行数少1 因为最后一行没/n
    freeze_vocab_size = 31029
    embedding_size = 200

    # nn网络通用配置
    dropout_prob = 0.5
    learning_rate = 1e-4
    num_classes = 2

    batch_size = 32
    num_epochs = 50

    print_per_batch = 100
    save_per_batch = 10

    # 数据路径
    data_dir = "../data/balance_data_jieba"
    train_filename = "train_cuts.txt"
    dev_filename = "dev_cuts.txt"

    # vocab位置
    pre_embedding_vocab_path = "pre_embedding_vocab.txt"  # tx词向量中有的
    oov_vocab_path = "oov_vocab.txt"  # 未登陆词词典

    #pre_embedding_table
    pre_embedding_table_name = "pre_embedding.txt"

    # cnn网络结构特定配置
    filter_sizes = [4]
    num_filters = 100
    fc_hidden_dim = 128



class TextCNN(object):

    def __init__(self, config):
        self.config = config

        # nn通用feed_dict的输入
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name="input_y")
        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    [self.config.freeze_vocab_size, self.config.embedding_size])
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.cnn()  # 如果不加self.cnn 代码不认识cnn中的self

    def cnn(self):
        with tf.device("/cpu:0"):
            embedding_freeze = tf.get_variable("freezez_embedding",
                                               [self.config.freeze_vocab_size, self.config.embedding_size],
                                               trainable=False)

            self.embedding_init = embedding_freeze.assign(self.embedding_placeholder)

            embedding_train = tf.get_variable("train_embedding",
                                              [self.config.train_vocab_size, self.config.embedding_size], trainable=True)
            embedding = tf.concat([embedding_freeze, embedding_train], axis=0)
            embedding_outputs = tf.nn.embedding_lookup(embedding, self.input_x)

        pooled_out = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("cnn-%s" % filter_size):
                conv = tf.layers.conv1d(embedding_outputs, self.config.num_filters, filter_size,
                                        name="conv-%s" % filter_size)
                max_pooling = tf.reduce_max(conv, 1, name="max_pooling%s" % filter_size)
                pooled_out.append(max_pooling)
        pooled_out = tf.concat(max_pooling, -1)
        with tf.name_scope("score"):
            fc = tf.layers.dense(pooled_out, self.config.fc_hidden_dim, name="fc1")
            fc = tf.layers.dropout(fc, self.config.dropout_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optm = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            self.true = tf.argmax(self.input_y, 1)
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            self.recall = tf.metrics.recall(labels=tf.argmax(self.input_y, 1),
                                            predictions=self.y_pred)