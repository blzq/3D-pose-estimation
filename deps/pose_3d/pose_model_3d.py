#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class PoseModel3d:
    def __init__(self, input_shape, n_outputs: int,
                 summary_dir='/tmp/tensorflow_logs',
                 saver_path='/tmp/tensorflow_ckpts/3dpose.ckpt',
                 restore_model=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0})
            self.sess = tf.Session(config=config)

            self.inputs = tf.placeholder(tf.float32, shape=input_shape)
            self.outputs = _build_model(self.inputs, n_outputs)

            self.sess.run(tf.global_variables_initializer())

            self.saver_path = saver_path
            self.saver = tf.train.Saver()
            if restore_model:
                try:
                    self.saver.restore(self.sess, self.saver_path)
                except tf.errors.InvalidArgumentError:
                    print("No model checkpoint found for given path {}"
                          .format(self.saver_path))
                    print("Continuing without loading model")

            self.writer = tf.summary.FileWriter(summary_dir)
            self.writer.add_graph(tf.get_default_graph())

    def save_model(self, save_model_path: str):
        tf.saved_model.simple_save(self.sess, save_model_path,
                                   {'model_in': self.inputs},
                                   {'model_out': self.outputs})

    def estimate(self, input_inst):
        with self.graph.as_default():
            out = \
                self.sess.run(self.outputs,
                              feed_dict={self.inputs: input_inst})
            return out

    def train(self, dataset, epochs: int, batch_size: int):
        with self.graph.as_default():
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            heatmaps, gt_pose, _ = iterator.get_next()

            loss = tf.losses.mean_squared_error(
                    labels=gt_pose, predictions=self.outputs)
            tf.summary.scalar('loss', loss)

            merged_summary = tf.summary.merge_all()

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train = optimizer.minimize(loss)

            for i in range(epochs):
                self.sess.run(iterator.initializer)
                while True:
                    try:
                        next_heatmap = self.sess.run(heatmaps)
                        _, summary = self.sess.run((train, merged_summary),
                                                    feed_dict={
                                                        self.inputs: next_heatmap})
                        self.writer.add_summary(summary, i)
                    except tf.errors.OutOfRangeError:
                        break
                if i % 10 == 0:
                    self.saver.save(self.sess, self.saver_path)


def _build_model(inputs, n_outputs: int):
    with tf.variable_scope('conv1'):
        conv1 = \
            tf.layers.conv2d(inputs, n_outputs, [3, 3])
        relu1 = tf.nn.relu(conv1)
    with tf.variable_scope('flatten1'):
        flatten1 = \
            tf.layers.flatten(relu1)
    with tf.variable_scope('linear1'):
        linear1 = tf.layers.dense(flatten1, n_outputs)
        relu2 = tf.nn.relu(linear1)
    return relu2
