#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os.path

import tensorflow as tf
from .network import build_model
from tf_smpl.batch_smpl import SMPL
from tf_rodrigues.rodrigues import rodrigues_batch

class PoseModel3d:
    def __init__(self, input_shape,
                 training=False,
                 summary_dir='/tmp/tensorflow_logs',
                 saver_path='/tmp/tensorflow_ckpts/3d_pose',
                 restore_model=True,
                 mesh_loss=False,
                 smpl_model=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # noqa
            self.sess = tf.Session(config=config)

            self.inputs = tf.placeholder(tf.float32, shape=input_shape,
                                         name='input_placeholder')
            self.outputs = build_model(self.inputs, training)

            self.saver_path = saver_path
            self.saver = tf.train.Saver()

            self.train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                                      self.graph)

            self.mesh_loss = mesh_loss
            if self.mesh_loss:
                self.smpl = SMPL(smpl_model)

            self.sess.run(tf.global_variables_initializer())

            if restore_model:
                ok_colour = '\033[92m'
                warn_colour = '\033[93m'
                normal_colour = '\033[0m'

                restore_path = os.path.dirname(self.saver_path)
                restore_ckpt = tf.train.latest_checkpoint(restore_path)
                if restore_ckpt != None:
                    try:
                        self.saver.restore(self.sess, restore_ckpt)
                        print("{}Model restored from checkpoint{}"
                                .format(ok_colour, normal_colour))
                    except:
                        print("{}Invalid model checkpoint found for given path {}"
                          .format(warn_colour, self.saver_path))
                        print("Continuing without loading model" + normal_colour)
                else:
                    print("{}No model checkpoint found for given path {}"
                          .format(warn_colour, self.saver_path))
                    print("Continuing without loading model" + normal_colour)

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
            dataset = dataset.shuffle(batch_size * 1000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            heatmaps, gt_pose, betas, frames = iterator.get_next()
            input_stack = tf.concat([heatmaps, frames], axis=3)
                # Not using image frames for the time being

            with tf.variable_scope("rodrigues"):
                out_mat = rodrigues_batch(self.outputs)
                gt_mat = rodrigues_batch(gt_pose)

            pose_loss = tf.losses.mean_squared_error(
                labels=gt_mat, predictions=out_mat)
            tf.summary.scalar('pose_loss', pose_loss)

            reg_loss = tf.losses.mean_squared_error(
                labels=tf.zeros(tf.shape(gt_pose)), predictions=self.outputs)
            scaled_reg_loss = reg_loss * 0.1
            tf.summary.scalar('reg_loss', scaled_reg_loss)

            total_loss = pose_loss + scaled_reg_loss

            if self.mesh_loss:
                output_meshes, _, _ = self.smpl(betas, self.outputs, get_skin=True)

                gt_meshes, _, _ = self.smpl(betas, gt_pose, get_skin=True)

                mesh_loss = tf.losses.mean_squared_error(
                    labels=gt_meshes, predictions=output_meshes)
                scaled_mesh_loss = mesh_loss * 2
                tf.summary.scalar('mesh_loss', scaled_mesh_loss)

                total_loss += scaled_mesh_loss

            tf.summary.scalar('total_loss', total_loss)
            merged_summary = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(total_loss)
            self.sess.run(tf.variables_initializer(optimizer.variables()))

            self.train_writer.add_graph(self.graph)

            for i in range(epochs):
                self.sess.run(iterator.initializer)
                while True:
                    try:
                        next_input = self.sess.run(input_stack)
                        feed = {self.inputs: next_input}
                        _, summary = self.sess.run((train, merged_summary),
                                                   feed_dict=feed)
                        self.train_writer.add_summary(summary, i)
                    except tf.errors.OutOfRangeError:
                        break
                if i % 10 == 0:
                    self.saver.save(self.sess, self.saver_path, global_step=i)

