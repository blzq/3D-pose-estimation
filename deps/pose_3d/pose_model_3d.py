#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class PoseModel3d:
    def __init__(self, summary_dir):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0})
            self.sess = tf.Session(config=config)
            self.inputs = tf.placeholder(tf.float32, shape=[None, 3])
            self.outputs = _linear_layer('linear1', self.inputs)
            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter(summary_dir)
            self.writer.add_graph(tf.get_default_graph())

    def estimate(self, input_inst):
        with self.graph.as_default():
            out = \
                self.sess.run(self.outputs,
                              feed_dict={self.inputs: input_inst})
            return out
    
    def train(self, x, y_true):
        with self.graph.as_default():
            loss = tf.losses.mean_squared_error(
                labels=y_true, predictions=self.outputs)
            tf.summary.scalar('loss', loss)

            merged_summary = tf.summary.merge_all()
            
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train = optimizer.minimize(loss)
            for i in range(20):
                _, summary = self.sess.run((train, merged_summary),
                                             feed_dict={self.inputs: x})
                self.writer.add_summary(summary, i)
    

def _linear_layer(var_scope: str, inputs):
    with tf.variable_scope(var_scope):
        linear = tf.layers.dense(inputs, 3)
    return linear