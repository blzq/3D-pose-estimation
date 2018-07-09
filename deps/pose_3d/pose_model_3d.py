#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from .network import build_model
from smpl_numpy.smpl_tf_class import SMPLModel

class PoseModel3d:
    def __init__(self, input_shape,
                 training=False,
                 summary_dir='/tmp/tensorflow_logs',
                 saver_path='/tmp/tensorflow_ckpts/3dpose.ckpt',
                 restore_model=True,
                 mesh_loss=False,
                 smpl_female=None,
                 smpl_male=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # noqa
            self.sess = tf.Session(config=config)

            self.inputs = tf.placeholder(tf.float32, shape=input_shape)
            self.outputs = build_model(self.inputs, training)

            self.saver_path = saver_path
            self.saver = tf.train.Saver()

            self.train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                                      self.graph)

            self.mesh_loss = mesh_loss
            if self.mesh_loss:
                self.smpl_female = SMPLModel(smpl_female)
                self.smpl_male = SMPLModel(smpl_male)

            self.sess.run(tf.global_variables_initializer())

            if restore_model:
                try:
                    self.saver.restore(self.sess, self.saver_path)
                except tf.errors.NotFoundError:
                    print("No model checkpoint found for given path {}"
                          .format(self.saver_path))
                    print("Continuing without loading model")


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
            heatmaps, gt_pose, shapes, _ = iterator.get_next()
                # Not using image frames for the time being

            pose_loss = tf.losses.mean_squared_error(
                labels=gt_pose, predictions=self.outputs)
            tf.summary.scalar('pose_loss', pose_loss)
            reg_loss = tf.losses.mean_squared_error(
                labels=tf.zeros(tf.shape(gt_pose)), predictions=self.outputs)
            scaled_reg_loss = reg_loss * 0.1
            tf.summary.scalar('reg_loss', scaled_reg_loss)
            total_loss = pose_loss + scaled_reg_loss

            if self.mesh_loss:
                betas = shapes
                trans = tf.zeros(self.smpl_female.trans_shape)

                get_inst_f = (lambda ob:
                    self.smpl_female.instance(ob[1], ob[0], trans, simplify=True))
                get_inst_m = (lambda ob:
                    self.smpl_male.instance(ob[1], ob[0], trans, simplify=True))

                outputs_f = tf.map_fn(get_inst_f, (self.outputs, betas), 
                    dtype=tf.float32, parallel_iterations=batch_size)
                outputs_m = tf.map_fn(get_inst_m, (self.outputs, betas), 
                    dtype=tf.float32, parallel_iterations=batch_size)
                outputs_avg = tf.add(outputs_f, outputs_m)

                gt_mesh_f = tf.map_fn(get_inst_f, (gt_pose, betas), 
                    dtype=tf.float32, parallel_iterations=batch_size)
                gt_mesh_m = tf.map_fn(get_inst_m, (gt_pose, betas), 
                    dtype=tf.float32, parallel_iterations=batch_size)
                gt_mesh_avg = tf.add(gt_mesh_f, gt_mesh_m)

                mesh_loss = tf.losses.mean_squared_error(
                    labels=gt_mesh_avg, predictions=outputs_avg)
                scaled_mesh_loss = mesh_loss * 5
                tf.summary.scalar('mesh_loss', scaled_mesh_loss)

                total_loss += scaled_mesh_loss

            tf.summary.scalar('total_loss', total_loss)
            merged_summary = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(total_loss)
            self.sess.run(tf.variables_initializer(optimizer.variables()))

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

