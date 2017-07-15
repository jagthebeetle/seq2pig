import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core


class RNNSearch(object):
    """Implementation of Bahdanau et al. 2014.

    Args:
        encoder_size: dimensionality of encoder state/output vectors.
        attention_size: dimensionality of attention kernel.
        decoder_size: dimensionality of decoder state/output vectors.
        input_shape: an int tuple of (batch_size, seq_length, feature_length).
        target_shape: an int tuple of (batch_size, max_seq_length,
            feature_length).
    """
    def __init__(self, encoder_size, attention_size, decoder_size,
                 input_shape, target_shape, dtype=tf.float32):
        with tf.variable_scope('encoder'):
            self.encoder = tf.nn.rnn_cell.GRUCell(encoder_size)

        with tf.variable_scope('decoder'):
            self.decoder = tf.nn.rnn_cell.GRUCell(decoder_size)

        self.attention_size = attention_size
        self.bahdanau = None
        self.decoder_cell = None
        self.projection_layer = core.Dense(self.targets.shape[2].value,
                                           activation=None,
                                           use_bias=True,
                                           name='projection')

        self.inputs = tf.placeholder(dtype, shape=input_shape, name='inputs')
        # Training
        with tf.variable_scope('training'):
            self.targets = tf.placeholder(
                dtype, shape=target_shape, name='targets')
            self.training_seq_lens = tf.placeholder(
                tf.int32, shape=(None,), name='training_seq_lens')

            self.sampling_rate = tf.placeholder(
                tf.float32, shape=(), name='previous_output_sampling_rate')
            self.train_op = None

        # Prediction
        self.predictions = None

        # Finish building graph
        self.build_model()
        self.build_train_op()
        self.build_predict_op()

    def build_model(self):
        encoder = self.encoder
        inputs = self.inputs
        with tf.variable_scope('encoder'):
            t_sequence = tf.unstack(inputs, axis=1,
                                    name='TimeMajorInputs')
            outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=encoder,
                                                           cell_bw=encoder,
                                                           inputs=t_sequence,
                                                           dtype=inputs.dtype)
        with tf.variable_scope('decoder'):
            with tf.name_scope('attention'):
                memory = tf.stack(outputs, axis=1,
                                  name='BatchMajorAnnotations')
                self.bahdanau = seq2seq.BahdanauAttention(
                    self.attention_size, memory=memory)

            raw_decoder = self.decoder
            decoder_cell = seq2seq.AttentionWrapper(raw_decoder, self.bahdanau,
                                                    output_attention=False)
            self.decoder_cell = decoder_cell

    def get_zero_state(self):
        batch_size, dtype = tf.shape(self.inputs)[0], self.inputs.dtype
        decoder_state0 = self.decoder.zero_state(batch_size, dtype)
        alignments = self.bahdanau(
            decoder_state0, self.bahdanau.initial_alignments(batch_size, dtype))
        expanded_alignments = tf.expand_dims(alignments, 1)
        attention_mechanism_values = self.bahdanau.values
        context = expanded_alignments @ attention_mechanism_values
        context1 = tf.squeeze(context, [1])
        t0 = tf.zeros([], dtype=tf.int32)
        return seq2seq.AttentionWrapperState(cell_state=decoder_state0, time=t0,
                                             alignments=alignments,
                                             alignment_history=(),
                                             attention=context1)

    def build_predict_op(self):
        with tf.variable_scope('predict'):
            decoder_cell = self.decoder_cell
            targets = self.targets
            sequence_lengths = self.training_seq_lens
            predict_helper = seq2seq.ScheduledOutputTrainingHelper(
                targets, sequence_lengths,
                sampling_probability=1.0,
                next_input_layer=self.projection_layer)
            decoder = seq2seq.BasicDecoder(decoder_cell, predict_helper,
                                           self.get_zero_state())
            output, _, _ = seq2seq.dynamic_decode(decoder,
                                                  output_time_major=True)

            self.predictions = self.projection_layer(output.rnn_output)

    def build_train_op(self):
        with tf.variable_scope('training'):
            decoder_cell = self.decoder_cell
            targets = self.targets
            sequence_lengths = self.training_seq_lens
            training_helper = seq2seq.ScheduledOutputTrainingHelper(
                targets, sequence_lengths,
                sampling_probability=self.sampling_rate,
                next_input_layer=self.projection_layer)
            decoder = seq2seq.BasicDecoder(
                decoder_cell, helper=training_helper,
                initial_state=self.get_zero_state())
            output, _, _ = seq2seq.dynamic_decode(decoder,
                                                  output_time_major=True)
            predictions = self.projection_layer.apply(output.rnn_output)
        time_major = tf.transpose(targets, perm=[1, 0, 2])
        x_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=time_major,
                                                            logits=predictions,
                                                            name='CrossEntropy')
        loss = tf.reduce_mean(x_entropy)
        self.train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), 0.001, 'Adam')
