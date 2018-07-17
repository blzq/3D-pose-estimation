#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os.path

import tensorflow as tf
from .network import build_model, build_discriminator
from . import config
from tf_smpl.batch_smpl import SMPL
from tf_perspective_projection.project import rodrigues_batch, project


class PoseModel3d:
    def __init__(self,
                 input_shape,
                 graph=None,
                 training=False,
                 train_dataset=None,
                 summary_dir='/tmp/tf_logs/3d_pose/',
                 saver_path='/tmp/tf_ckpts/3d_pose/ckpts/3d_pose.ckpt',
                 restore_model=True,
                 reproject_loss=True,
                 mesh_loss=False,
                 smpl_model=None,
                 discriminator=False):
        self.graph = graph if graph != None else tf.get_default_graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # noqa
            self.sess = tf.Session(config=config)

            self.discriminator = discriminator
            if training:
                self.dataset = train_dataset
                self.input_handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                    self.input_handle,
                    self.dataset.output_types, self.dataset.output_shapes)
                self.next_input = iterator.get_next()
                # placeholders for shape inference
                self.input_placeholder = tf.placeholder_with_default(
                    self.next_input[0], input_shape)
                self.input_placeholder_loc = tf.placeholder_with_default(
                    self.next_input[3], [None, 18, 2])
                self.outputs = build_model(
                    self.input_placeholder, self.input_placeholder_loc, training)
                if self.discriminator:
                    combin_in = tf.concat(
                        [self.next_input[1], self.outputs], 0)
                    self.discrim_outputs = build_discriminator(combin_in)
            else:
                self.input_placeholder = tf.placeholder(tf.float32, input_shape)
                self.input_placeholder_loc = tf.placeholder(
                    tf.float32, [None, 18, 2])
                self.outputs = build_model(
                    self.input_placeholder, self.input_placeholder_loc, training)

            self.saver_path = saver_path
            self.saver = None

            subdir = 'train' if training else 'test'
            self.summary_writer = tf.summary.FileWriter(
                os.path.join(summary_dir, subdir), self.graph)

            self.mesh_loss = mesh_loss
            self.reproject_loss = reproject_loss
            if self.mesh_loss or self.reproject_loss:
                self.smpl = SMPL(smpl_model)

            self.sess.run(tf.global_variables_initializer())

            self.restore = restore_model
            self.already_restored = False

    def save_model(self, save_model_path: str):
        with self.graph.as_default():
            tf.saved_model.simple_save(self.sess, save_model_path,
                                       {'in': self.input_placeholder,
                                        'in_loc': self.input_placeholder_loc},
                                       {'out': self.outputs})

    def restore_from_checkpoint(self):
        with self.graph.as_default():
            if self.already_restored:
                return
            # terminal colours for printing
            ok_col, warn_col, normal_col = '\033[92m', '\033[93m', '\033[0m'

            restore_path = os.path.dirname(self.saver_path)
            restore_ckpt = tf.train.latest_checkpoint(restore_path)
            if restore_ckpt != None:
                try:
                    self.saver.restore(self.sess, restore_ckpt)
                    print(
                        "{}Model restored from checkpoint at {}{}".format(
                        ok_col, self.saver_path, normal_col))
                    self.already_restored = True
                except:
                    print(
                        "{}Invalid model checkpoint found for given path {}"
                        "\nContinuing without loading model{}".format(
                        warn_col, self.saver_path, normal_col))
            else:
                print("{}No model checkpoint found for given path {}"
                        "\nContinuing without loading model{}".format(
                        warn_col, self.saver_path, normal_col))

    def estimate(self, input_inst, input_loc):
        with self.graph.as_default():
            if self.saver is None:
                self.saver = tf.train.Saver()
            if self.restore:
                self.restore_from_checkpoint()
            summary = tf.summary.merge_all()
            out, summary_val = self.sess.run(
                (self.outputs, summary),
                feed_dict={self.input_placeholder: input_inst,
                           self.input_placeholder_loc: input_loc})
            self.summary_writer.add_summary(summary_val)
        return out

    def train(self, batch_size: int, epochs: int):
        with self.graph.as_default():
            self.dataset = self.dataset.shuffle(batch_size * 10)
            self.dataset = self.dataset.batch(batch_size)
            iterator = self.dataset.make_initializable_iterator()
            train_handle = self.sess.run(iterator.string_handle())

            _, gt_pose, betas, gt_joints2d = self.next_input

            out_pose = self.outputs[:, :72]

            with tf.variable_scope("rodrigues"):
                out_mat = rodrigues_batch(tf.reshape(out_pose, [-1, 3]))
                gt_mat = rodrigues_batch(tf.reshape(gt_pose, [-1, 3]))

            pose_loss = tf.losses.mean_squared_error(
                labels=gt_mat, predictions=out_mat)
            tf.summary.scalar('pose_loss', pose_loss, family='losses')

            reg_loss = tf.losses.mean_squared_error(
                labels=tf.zeros(tf.shape(gt_pose)), predictions=out_pose)
            scaled_reg_loss = reg_loss * config.reg_loss_scale
            tf.summary.scalar('reg_loss', scaled_reg_loss, family='losses')
            total_loss = pose_loss + scaled_reg_loss

            if self.mesh_loss or self.reproject_loss:
                out_meshes, out_joints, _ = self.smpl(betas, out_pose, 
                                                      get_skin=True)
                gt_meshes, _, _ = self.smpl(betas, gt_pose, get_skin=True)

            if self.mesh_loss:
                mesh_loss = tf.losses.mean_squared_error(
                    labels=gt_meshes, predictions=out_meshes)
                scaled_mesh_loss = mesh_loss * config.mesh_loss_scale
                tf.summary.scalar('mesh_loss', scaled_mesh_loss,
                                  family='losses')
                total_loss += scaled_mesh_loss

            if self.reproject_loss:
                out_cam_pos = self.outputs[:, 72:75]
                out_cam_rot = self.outputs[:, 75:78]
                out_cam_foc = self.outputs[:, 78]
                # out_joints reshape from (batch, j, 3) to (batch * j, 3)
                out_2d = project(tf.reshape(out_joints, [-1, 3]), 
                                 out_cam_pos, out_cam_rot, out_cam_foc)
                out_2d = tf.gather(out_2d, [1, 0], axis=1)
                # gt_joints2d reshape from (batch, j, 2) to (batch * j, 2)
                reproj_loss = tf.losses.mean_squared_error(
                    out_2d, tf.reshape(gt_joints, [-1, 2])
                scaled_reproj_loss = reproj_loss * config.reproj_loss_scale
                tf.summary.scalar('reprojection_loss', scaled_reproj_loss,
                                  family='losses')
                total_loss += scaled_reproj_loss

            if self.discriminator:
                discrim_real_out, discrim_pred_out = \
                    tf.split(self.discrim_outputs, 2, axis=0)

                disc_real_loss = tf.losses.mean_squared_error(
                    discrim_real_out, tf.ones(tf.shape(gt_pose)))
                tf.summary.scalar('discriminator_real_loss', disc_real_loss,
                                  family='discriminator')
                disc_fake_loss = tf.losses.mean_squared_error(
                    discrim_pred_out, tf.zeros(tf.shape(out_pose)))
                tf.summary.scalar('discriminator_fake_loss', disc_fake_loss,
                                  family='discriminator')
                disc_enc_loss = tf.losses.mean_squared_error(
                    discrim_pred_out, tf.ones(tf.shape(out_pose)))
                disc_enc_scaled_loss = disc_enc_loss * config.disc_loss_scale
                tf.summary.scalar('discriminator_loss', disc_enc_scaled_loss,
                                  family='losses')
                disc_optimizer = tf.train.AdamOptimizer()
                discriminator_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                train_d_real = disc_optimizer.minimize(
                    disc_real_loss, var_list=discriminator_vars)
                train_d_fake = disc_optimizer.minimize(
                    disc_fake_loss, var_list=discriminator_vars)
                self.sess.run(tf.variables_initializer(
                    disc_optimizer.variables()))
                total_loss += disc_enc_scaled_loss

            tf.summary.scalar('total_loss', total_loss, family='losses')
            summary = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer()
            encoder_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
            train = optimizer.minimize(total_loss, var_list=encoder_vars)
            self.sess.run(tf.variables_initializer(optimizer.variables()))

            if self.saver is None:
                self.saver = tf.train.Saver()
            if self.restore:
                self.restore_from_checkpoint()

            self.summary_writer.add_graph(self.graph)

            for i in range(epochs):
                self.sess.run(iterator.initializer)
                feed = {self.input_handle: train_handle}
                j = 0
                while True:
                    try:
                        if self.discriminator:
                            _, summary_eval, _, _ = self.sess.run(
                                (train, summary, train_d_real, train_d_fake),
                                feed_dict=feed)
                        else:
                            _, summary_eval = self.sess.run(
                                (train, summary),
                                feed_dict=feed)
                        self.summary_writer.add_summary(summary_eval, i)
                    except tf.errors.OutOfRangeError:
                        break
                    if j % 1000 == 0:
                        self.saver.save(self.sess,
                                        self.saver_path, global_step=i)
                    j += 1
                self.saver.save(self.sess, self.saver_path, global_step=i)
