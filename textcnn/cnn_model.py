import tensorflow as tf


class CNNConfig(object):

    #nlp通用配置
    max_seq_len = 10
    vocab_size = 40000
    embedding_dim = 64
    #nn网络通用配置
    dropout_prob = 0.5
    learning_rate = 1e-4
    num_classes = 10
    batch_size = 64
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10

    #cnn网络结构特定配置
    filter_size = [5]
    num_filters = 100
    fc_hidden_dim = 128



class TextCNN(object):

    def __init__(self, config):
        self.config = config

        #nn通用feed_dict的输入
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_seq_len], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name = "input_y")
        self.dropout = tf.placeholder(tf.float32, name = "dropout")

        self.cnn()  #如果不加self.cnn 代码不认识cnn中的self

    def cnn(self):

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_outputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_outputs, self.config.num_filters, self.config.filter_size, name = "conv")
            max_pooling = tf.reduce_max(conv, 1, name="max_pooling")

        with tf.name_scope("score"):
            fc = tf.layers.dense(max_pooling, self.config.fc_hidden_dim, name = "fc1")
            fc = tf.layers.dropout(fc, self.config.dropout_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name = "fc2")
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optm = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

