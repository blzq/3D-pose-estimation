#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from .network import build_model

class PoseModel3d:
    def __init__(self, input_shape, n_outputs: int,
                 training=False,
                 summary_dir='/tmp/tensorflow_logs',
                 saver_path='/tmp/tensorflow_ckpts/3dpose.ckpt',
                 restore_model=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            self.inputs = tf.placeholder(tf.float32, shape=input_shape)
            self.outputs = build_model(self.inputs, training)

            self.saver_path = saver_path
            self.saver = tf.train.Saver()
            if restore_model:
                try:
                    self.saver.restore(self.sess, self.saver_path)
                except tf.errors.NotFoundError:
                    print("No model checkpoint found for given path {}"
                          .format(self.saver_path))
                    print("Continuing without loading model")

            self.train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                                      self.graph)

            self.sess.run(tf.global_variables_initializer())

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
            dataset = dataset.shuffle(128)
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
                        feed = {self.inputs: next_heatmap}
                        _, summary = self.sess.run((train, merged_summary),
                                                   feed_dict=feed)
                        self.train_writer.add_summary(summary, i)
                    except tf.errors.OutOfRangeError:
                        break
                if i % 10 == 0:
                    self.saver.save(self.sess, self.saver_path)

