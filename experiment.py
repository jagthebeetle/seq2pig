import datetime
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import flags

import model
import piglatin as pig

flags.DEFINE_integer('batch_size', 50, 'Mini-batch size.')
FLAGS = flags.FLAGS


def vec(c):
    """One hot encoding of characters."""
    nth_c = ord(c) - 97
    vector = [0]*26
    vector[nth_c] = 1
    return vector


def print_batch(predictions, time_major=False):
    """Grab max indices against last axis and turn to chars."""
    char_of_int_fn = np.vectorize(lambda c: chr(c + 97))
    char_indices = np.argmax(predictions, axis=2)
    if time_major:
        char_indices = char_indices.T
    chars = char_of_int_fn(char_indices)
    print('\n'.join([''.join(w) for w in chars]))


def get_training_inputs(words_df, batch_size=2):
    """Returns tuple of inputs, targets, and sequence_lengths."""
    words_column = words_df.columns[0]
    samples = words_df.sample(batch_size, replace=True)[words_column].values
    original_chars, pig_latin_chars = [], []
    for word in samples:
        original_chars.append([vec(c) for c in word])
        pig_latin_chars.append([vec(c) for c in pig.translate(word)])
    pig_latin_padded = [char_seq + [[0]*26]*(9-len(char_seq))
                        for char_seq in pig_latin_chars]
    # TODO(bug): How to feed correct sequence lengths?
    seq_lens = [len(char_seq) for char_seq in pig_latin_padded]
    return original_chars, pig_latin_padded, seq_lens


def get_log_dir():
    return path.join('tf_logs', datetime.datetime.now().strftime('%H%M'))


words = pd.read_csv('words.csv')
word_len_mask = words['word'].str.len() == 6
words = words.loc[word_len_mask]


def main(unused_argv):
    batch_size = FLAGS.batch_size
    rnn_search = model.RNNSearch(60, 60, 20,
                                 input_shape=(batch_size, 6, 26),
                                 target_shape=(batch_size, 9, 26))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(30000):
            inputs, targets, seq_lens = get_training_inputs(words, batch_size)
            prev_output_sampling_rate = min(i**1.02/30000, 1)
            feed = {
                rnn_search.inputs: inputs,
                rnn_search.targets: targets,
                rnn_search.training_seq_lens: seq_lens,
                rnn_search.sampling_rate: prev_output_sampling_rate
            }

            if i % 200 == 0:
                loss = sess.run(rnn_search.train_op, feed_dict=feed)
                print('loss: {} at step {}, sampled {}'.format(
                      loss, i, prev_output_sampling_rate))
            else:
                sess.run(rnn_search.train_op, feed_dict=feed)

        for i in range(2):
            inputs, targets, seq_lens = get_training_inputs(words, 50)
            feed = {
                rnn_search.inputs: inputs,
                rnn_search.targets: targets,
                rnn_search.training_seq_lens: seq_lens
            }

            predictions = sess.run(rnn_search.predictions, feed_dict=feed)
            print_batch(inputs)
            print_batch(predictions, time_major=True)


if __name__ == '__main__':
    tf.app.run()
