import tensorflow as tf
# embedding_outputs = tf.Variable(tf.random_normal([2, 10,4], stddev=1, seed=1))
#
# conv = tf.layers.conv1d(embedding_outputs, 3, 3, padding="same",name="conv")
# maxpooling = tf.reduce_max(conv, 1, name="max_pooling")
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(embedding_outputs))
#     print(sess.run(conv))
#     print(sess.run(maxpooling))
#
#
tf.get_variable