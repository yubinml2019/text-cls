import tensorflow as tf


class FastTextConfig(object):
    # nlp通用配置
    max_seq_len = 10
    vocab_size = 40000
    embedding_dim = 64
    # nn网络通用配置
    dropout_prob = 0.5
    learning_rate = 1e-4
    num_classes = 10
    batch_size = 64
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10

    # fasttext网络结构特定配置
    fc_hidden_dim = 128


class Fasttext(object):

    def __init__(self, config):
        self.config = config

        # nn通用feed_dict的输入
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name="input_y")
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = tf.constant(0.01)
        self.fasttext()

    def fasttext(self):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_outputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("fasttext"):
            sentence_embeddings = tf.reduce_mean(embedding_outputs, axis=1)
        with tf.name_scope("drop_out"):
            drop_out = tf.layers.dropout(sentence_embeddings, self.config.dropout_prob)

            # fasttext 有没有激活函数的使用啊？？
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.config.embedding_dim, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(drop_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) + self.l2_reg_lambda * self.l2_loss

            self.optm = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
