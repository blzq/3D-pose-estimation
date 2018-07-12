#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os.path

import tensorflow as tf
from .network import build_model, build_discriminator
from tf_smpl.batch_smpl import SMPL
from tf_rodrigues.rodrigues import rodrigues_batch

class PoseModel3d:
    def __init__(self, input_shape,
                 training=False,
                 summary_dir='/tmp/tensorflow_logs',
                 saver_path='/tmp/tensorflow_ckpts/3d_pose',
                 restore_model=True,
                 mesh_loss=False,
                 smpl_model=None,
                 discriminator=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # noqa
            self.sess = tf.Session(config=config)

            self.input_handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                self.input_handle, tf.float32, input_shape)
            next_input = iterator.get_next()[0]
            self.outputs = build_model(next_input, training)
            
            self.discriminator = discriminator
            if self.discriminator:
                self.d_input_handle = tf.placeholder(tf.string, shape=[])
                discrim_iterator = tf.data.Iterator.from_string_handle(
                    self.d_input_handle, tf.float32, [None, 72])
                next_d_input = discrim_iterator.get_next()
                combined_in = tf.concat([self.outputs, next_d_input], 0)
                self.discrim_outputs = build_discriminator(combined_in)

            self.saver_path = saver_path
            self.saver = tf.train.Saver()

            subdir = 'train' if training else 'test'
            self.summary_writer = tf.summary.FileWriter(
                os.path.join(summary_dir, subdir), self.graph)

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
                        print(
                            "{}Model restored from checkpoint at {}{}".format(
                            ok_colour, self.saver_path, normal_colour))
                    except:
                        print(
                          "{}Invalid model checkpoint found for given path {}"
                          "\nContinuing without loading model{}".format(
                          warn_colour, self.saver_path, normal_colour))

                else:
                    print("{}No model checkpoint found for given path {}"
                          "\nContinuing without loading model{}".format(
                          warn_colour, self.saver_path, normal_colour))

    def save_model(self, save_model_path: str):
        tf.saved_model.simple_save(self.sess, save_model_path,
                                   {'model_in': self.inputs},
                                   {'model_out': self.outputs})

    def estimate(self, input_inst):
        with self.graph.as_default():
            merged_summary = tf.summary.merge_all()
            out, summary = self.sess.run((self.outputs, merged_summary),
                                         feed_dict={self.inputs: input_inst})
            self.summary_writer.add_summary(summary)
        return out

    def train(self, dataset, epochs: int, batch_size: int):
        with self.graph.as_default():
            dataset = dataset.shuffle(batch_size * 10)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            heatmaps, gt_pose, betas, frames = iterator.get_next()

            # tensorboard visualise inputs
            # reduced_heatmaps = tf.reduce_sum(
            #     heatmaps[:, :, :, :18], axis=3, keepdims=True)
            # tf.summary.image('heatmaps', reduced_heatmaps)
            # tf.summary.image('input_image', frames)

            input_stack = tf.concat([heatmaps, frames], axis=3)
            

            with tf.variable_scope("rodrigues"):
                out_mat = rodrigues_batch(tf.reshape(self.outputs, [-1, 3]))
                gt_mat = rodrigues_batch(tf.reshape(gt_pose, [-1, 3]))            

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
            
            if self.discriminator:
                discrim_real_out, discrim_pred_out = \
                    tf.split(self.discrim_outputs, 2, axis=0)

                disc_real_loss = tf.losses.mean_squared_error(
                    discrim_real_out, tf.ones(tf.shape(gt_pose)))
                disc_fake_loss = tf.losses.mean_squared_error(
                    discrim_pred_out, tf.zeros(tf.shape(self.outputs)))
                disc_enc_loss = tf.losses.mean_squared_error(
                    discrim_pred_out, tf.ones(tf.shape(self.outputs)))
                tf.summary.scalar('discriminator_loss', disc_enc_loss)
                disc_optimizer = tf.train.AdamOptimizer()
                self.sess.run(tf.variables_initializer(
                    disc_optimizer.variables()))
                train_d_real = disc_optimizer.minimize(disc_real_loss)
                train_d_fake = disc_optimizer.minimize(disc_fake_loss)
                total_loss += disc_enc_loss

            tf.summary.scalar('total_loss', total_loss)
            summary = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(total_loss)
            self.sess.run(tf.variables_initializer(optimizer.variables()))

            self.summary_writer.add_graph(self.graph)

            for i in range(epochs):
                self.sess.run(iterator.initializer)
                while True:
                    try:
                        next_input = self.sess.run(input_stack)
                        feed = {self.inputs: next_input}
                        if self.discriminator:
                            next_gt = self.sess.run(gt_pose)
                            feed[self.discrim_in] = next_gt

                        if self.discriminator:
                            _, _, _, summary_val = self.sess.run(
                                (train, train_d_fake, train_d_real, summary), 
                                feed_dict=feed)
                        else:
                            _, summary_val = self.sess.run(
                                train, summary)
                        self.summary_writer.add_summary(summary_val, i)
                    except tf.errors.OutOfRangeError:
                        break
                self.saver.save(self.sess, self.saver_path, global_step=i)

