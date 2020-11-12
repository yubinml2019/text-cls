import tensorflow as tf
import re
class LSTMConfig(object):
    max_seq_len = 500
    vocab_size = 32492
    #train_vocab_size = 1463
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

    # lstm_sizes
    embedding_dim = 256
    use_bilstm = 1
    use_basic_cell = 1
    use_bidirectional = 1

class LSTM(object):
    def __init__(self, config):
        tf.set_random_seed(66)
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_seq_len], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name="input_y")
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding,
                                                      self.input_x)  # batch_size * max_seq_len * embedding_dim
            embedding_inputs = tf.nn.dropout(embedding_inputs, self.dropout)

        with tf.name_scope("lstm"):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.embedding_dim, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.embedding_dim, forget_bias=1.0, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                             lstm_bw_cell,
                                                                             embedding_inputs,
                                                                             dtype=tf.float32,
                                                                             time_major=False,
                                                                             scope=None)
            bilstm_out = tf.concat([output_fw, output_bw], axis=2)
            bilstm_out = tf.reduce_mean(bilstm_out, axis=1)

        with tf.name_scope("dropout"):
            rnn_drop = tf.nn.dropout(bilstm_out, self.config.dropout_prob)

        with tf.name_scope("score"):
            # fc = tf.layers.dense(bilstm_out, self.embedding_dim, activation=tf.nn.relu, name='fc1') # batch_size * hidden_dim
            # fc_drop = tf.layers.dropout(fc, self.keep_prob)

            # classify
            self.logits = tf.layers.dense(rnn_drop, self.config.num_classes, name='fc2')  # batch_size * num_classes
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)

            l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.loss += l2_loss

            # optim
            self.optm = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
            with tf.name_scope("accuracy"):
                # 准确率
                correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
                self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")

                self.recall = tf.metrics.recall(labels=tf.argmax(self.input_y, 1),
                                            predictions=self.y_pred)
