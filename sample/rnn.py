"""Recurrent Neural Network with TensorFlow Exemplo"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

num_epochs = 3
num_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    weights = tf.Variable(tf.random_normal([rnn_size,num_classes]))
    biases = tf.Variable(tf.random_normal([num_classes]))

    layer = {'weights':weights,
             'biases':biases}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            msg = 'Epoch {} completed out of {} loss: {}' \
                .format(epoch, num_epochs, epoch_loss)
                                                                
            print(msg)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        data = {x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), 
                y: mnist.test.labels}
        accuracy_value = accuracy.eval(data)

        print('Accuracy:', accuracy_value)


if __name__ == '__main__':
    train_neural_network(x)