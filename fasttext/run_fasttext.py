import os
from fasttext_model import *
from utils import cut_files, build_vocab, read_category, read_vocab, encode_file, batch_iter

model_dir = "./checkpoint"
data_dir = "../data/THUCNews/processed"

vocab_path = os.path.join(data_dir, "vocab.txt")


def train():
    # save = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    global_steps = 0
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for epoch in range(config.num_epochs):
        print("Epoch: {}".format(epoch + 1))
        batch_data = batch_iter(x_train, y_train, config.batch_size)
        for batch_x, batch_y in batch_data:
            feed_dict = feed_data(batch_x, batch_y, config.dropout_prob)

            if global_steps % config.print_per_batch == 0:
                feed_dict[model.dropout] = 1.0
                train_acc, train_loss = session.run([model.acc, model.loss], feed_dict=feed_dict)
            message = "train acc: {0}"
            print(message.format(train_acc))

            feed_dict[model.dropout] = config.dropout_prob

            session.run(model.optm, feed_dict)
            global_steps += 1


def feed_data(batch_x, batch_y, dropout_prob):
    feed_dict = {
        model.input_x: batch_x,
        model.input_y: batch_y,
        model.dropout: dropout_prob
    }
    return feed_dict


if __name__ == "__main__":

    print("starting cnn")
    config = FastTextConfig()  # 这个config为什么在train函数中可以用？？

    train_path = os.path.join(data_dir, "train_cuts.txt")
    val_path = os.path.join(data_dir, "dev_cuts.txt")
    if not os.path.exists(vocab_path):
        build_vocab(train_path, vocab_path, config.vocab_size)

    cate2id = read_category(train_path)
    word2id = read_vocab(vocab_path)

    x_train, y_train = encode_file(train_path, word2id, cate2id)
    x_val, y_val = encode_file(val_path, word2id, cate2id)

    model = Fasttext(config)
    train()
