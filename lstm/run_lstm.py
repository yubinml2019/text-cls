import numpy as np
import tensorflow as tf
from sklearn import metrics
import os
import time
from lstm_model import *
from utils import *


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, config.batch_size, False)
    total_loss = 0.0
    total_acc = 0.0
    total_recall = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc, recall = sess.run([model.loss, model.acc, model.recall], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
        total_recall += recall[0] * batch_len

    return total_loss / data_len, total_acc / data_len, total_recall / data_len


def feed_data(batch_x, batch_y, dropout_prob):
    feed_dict = {
        model.input_x: batch_x,
        model.input_y: batch_y,
        model.dropout: dropout_prob
    }
    return feed_dict



def train():
    # save = tf.train.Saver()

    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    global_steps = 0
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    tensorboard_dir = 'tensorboard/'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    for file in os.listdir(tensorboard_dir):
        file2r = os.path.join(tensorboard_dir, file)
        if os.path.isfile(file2r):
            os.remove(file2r)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    session.run(tf.local_variables_initializer())

    train_writer = tf.summary.FileWriter(tensorboard_dir + "train", session.graph)

    train_writer.add_graph(session.graph)

    for epoch in range(config.num_epochs):
        print("Epoch: {}".format(epoch + 1))
        batch_data = batch_iter(x_train, y_train, config.batch_size)
        for batch_x, batch_y in batch_data:
            feed_dict = feed_data(batch_x, batch_y, config.dropout_prob)

            if global_steps % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                train_writer.add_summary(s, global_steps)

            if global_steps % config.print_per_batch == 0:
                feed_dict[model.dropout] = 1.0
                train_loss, train_acc, train_recall = session.run([model.loss, model.acc, model.recall],
                                                                  feed_dict=feed_dict)
                eval_loss, eval_acc, eval_recall = evaluate(session, x_val, y_val)

                message = "train loss: {0} acc: {1} recall: {2} eval loss:{3}, acc: {4}, recall: {5}"
                print(message.format(train_loss, train_acc, train_recall[0], eval_loss, eval_acc, eval_recall))

            feed_dict[model.dropout] = config.dropout_prob
            session.run(model.optm, feed_dict)
            global_steps += 1
    #predicts, labels = print_test_log(session, x_val, y_val)
    #check_test_data(val_path, predicts, labels, id2cate)

if __name__ == "__main__":

    model_dir = "./checkpoint"
    data_dir = "../data/balance_data_jieba/"

    vocab_path = os.path.join(data_dir, "vocab.txt")

    print("starting bilstm")
    config = LSTMConfig()

    train_path = os.path.join(data_dir, "train_cuts.txt")
    val_path = os.path.join(data_dir, "dev_cuts.txt")
    if not os.path.exists(vocab_path):
        build_vocab(train_path, vocab_path, config.vocab_size)

    cate2id = read_category(train_path)
    word2id = read_vocab(vocab_path)

    x_train, y_train = encode_file(train_path, word2id, cate2id, config.max_seq_len)
    x_val, y_val = encode_file(val_path, word2id, cate2id, config.max_seq_len)
    model = LSTM(config)
    train()
