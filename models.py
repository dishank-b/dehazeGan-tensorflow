# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
import glob
import time

class dehazeGan(object):
	"""My own Dehaze GAN"""
	def __init__(self, model_path):
		self.model_path = model_path
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"
		if not os.path.exists(model_path):
			os.makedirs(self.graph_path+"train/")
			os.makedirs(self.graph_path+"val/")
		
	def _debug_info(self):
		variables_names = [[v.name, v.get_shape().as_list()] for v in tf.trainable_variables()]
		print "Trainable Variables:"
		tot_params = 0
		gen_param = 0
		for i in variables_names:
			var_params = np.prod(np.array(i[1]))
			tot_params += var_params
			print i[0], i[1], var_params
			if "generator" in i[0]:
				gen_param+=var_params
		print "Total number of Trainable Parameters: ", str(tot_params/1000.0)+"K"
		print "Total number of Generator Parameters: ", str(gen_param/1000.0)+"K"
		return str(tot_params/1000.0)+"K", str(gen_param/1000.0)+"K"


	def _generator(self, input_img):
		with tf.variable_scope("generator") as scope:
			with tf.variable_scope("TransMap") as scope:
				trans_conv1 = conv_2d(input_img, output_chan= 10, kernel=[3, 3], stride=[2, 2], use_bn=False, activation=leaky_relu,
					train_phase=self.train_phase, name="transconv1")
				print trans_conv1.get_shape()
				trans_conv2 = conv_2d(trans_conv1, output_chan= 10, kernel=[3, 3], stride=[3, 3], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="transconv2")
				print trans_conv2.get_shape()
				trans_conv3 = conv_2d(unpool(trans_conv2, [3,3], "upsample1"), output_chan= 10, kernel=[3, 3], stride=[1, 1], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="transconv3")
				print trans_conv3.get_shape()
				trans_conv4 = conv_2d(unpool(trans_conv3,[2,2] ,"upsample2"), output_chan= 1, kernel=[3, 3], stride=[1, 1], use_bn=False, activation=tf.tanh,
					train_phase=self.train_phase, name="transconv4")
				trans_conv4 = trans_conv4[:,:,2:-2,:]
				
			with tf.variable_scope("clear_image") as scope:
				conv1 = conv_2d(input_img, output_chan=16, kernel=[3,3], stride=[1,1], use_bn=False, activation=leaky_relu,
					train_phase=self.train_phase, name="conv1")
				print conv1.get_shape()
				conv2 = conv_2d(conv1, output_chan= 32, kernel=[3, 3], stride=[2, 2], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="conv2")
				print conv2.get_shape()
				conv3 = conv_2d(conv2, output_chan= 64, kernel=[3, 3], stride=[3, 3], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="conv3")
				print conv3.get_shape()
				conv4 = conv_2d(conv2, output_chan= 64, kernel=[5, 5], stride=[3, 3], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="conv4")
				print conv4.get_shape()
				in_concat = tf.concat([conv3, conv4, trans_conv2], axis=3,name="concat1")
				upsample1 = unpool(in_concat, [3,3], "upsample1")
				conv5 = conv_2d(upsample1, output_chan=32, kernel=[3,3], stride=[1,1], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="conv5")
				print conv5.get_shape()
				conv2_pad = tf.pad(conv2, [[0, 0], [0, 0], [1, 1], [0, 0]], "SYMMETRIC")
				in_concat = tf.concat([conv2_pad, conv5, trans_conv3], axis=3, name="concat2")
				upsample2 = unpool(in_concat, [2,2], "upsample2")
				conv6 = conv_2d(upsample2, output_chan=16, kernel=[3,3], stride=[1, 1], use_bn=True, activation=leaky_relu,
					train_phase=self.train_phase, name="conv6")
				print conv6.get_shape()
				print trans_conv4[:,:,2:-2,:].get_shape()
				in_concat = tf.concat([conv1, conv6[:,:,2:-2,:], trans_conv4], axis=3, name="concat3")
				conv7 = conv_2d(in_concat, output_chan=3, kernel=[3,3], stride=[1,1], use_bn=False, activation=tf.tanh,
					train_phase=self.train_phase, name="conv7")
				print conv7.get_shape()
			
		return conv7, trans_conv4

	def _discriminator(self, gen_input, gen_output, reuse):
		with tf.variable_scope("discriminator", reuse=reuse) as scope:
			conv1 = conv_2d(tf.concat([gen_input, gen_output],3), output_chan=16, kernel=[3,3], stride=[2,2], use_bn=False, activation=leaky_relu, 
				train_phase=self.train_phase, name="conv1")
			conv2 = conv_2d(conv1, output_chan=32, kernel=[3,3], stride=[2,2], use_bn=True, activation=leaky_relu, 
				train_phase=self.train_phase, name="conv2")
			conv3 = conv_2d(conv2, output_chan=64, kernel=[3,3], stride=[2,2], use_bn=True, activation=leaky_relu, 
				train_phase=self.train_phase, name="conv3")
			conv4 = conv_2d(conv3, output_chan=128, kernel=[3,3], stride=[2,2], use_bn=True, activation=leaky_relu, 
				train_phase=self.train_phase, name="conv4")
			conv5 = conv_2d(conv4, output_chan=1, kernel=[3,3], stride=[1,1], use_bn=False, activation=tf.sigmoid, 
				train_phase=self.train_phase, name="conv5")

		print conv5.get_shape()
		return conv5

	def build_model(self):
		with tf.name_scope("Inputs") as scope:
			self.haze_in = tf.placeholder(tf.float32, shape=[None,240,320,3], name="Haze_Image")
			self.clear_in = tf.placeholder(tf.float32, shape=[None,240,320,3], name="Clear_Image")
			self.trans_in = tf.placeholder(tf.float32, shape=[None,240,320,1], name="Trans_Map")
			self.train_phase = tf.placeholder(tf.bool, name="is_training")
			hazy_summ = tf.summary.image("Hazy_image", (self.haze_in+1)/2)
			clear_summ = tf.summary.image("clear_in", (self.clear_in+1)/2)

		with tf.name_scope("Model") as scope:
			self.gen_out, self.trans_out = self._generator(self.haze_in)
			self.dis_real = self._discriminator(self.haze_in, self.clear_in, reuse=False)
			self.dis_fake = self._discriminator(self.haze_in, self.gen_out, reuse=True)
			gen_out_summ = tf.summary.image("output_image", (self.gen_out+1)/2)

		with tf.name_scope("Loss") as scope:
			self.dis_loss = tf.reduce_mean(-(tf.log(self.dis_real + 1e-12) + tf.log(1 - self.dis_fake + 1e-12)), name="dis_loss")
			self.gen_ad_loss = tf.reduce_mean(-tf.log(self.dis_fake + 1e-12), name="gen_ad_loss")
			self.gen_L1_loss = tf.reduce_mean(tf.abs(self.clear_in - self.gen_out), name="gen_L1_loss")
			self.trans_loss = tf.losses.mean_squared_error(self.trans_out, self.trans_in)
			self.gen_loss = self.gen_ad_loss+100.0*(self.gen_L1_loss+self.trans_loss)

			dis_loss_summ = tf.summary.scalar("dis_loss", self.dis_loss)
			gen_l1_summ = tf.summary.scalar("gen_l1", self.gen_L1_loss)
			gen_ad_summ = tf.summary.scalar("gen_ad_loss", self.gen_ad_loss)
			trans_summ = tf.summary.scalar("trans_loss", self.trans_loss)
							 
		with tf.name_scope("Optimizers") as scope:
			dis_var_list = [var for var in tf.trainable_variables() if "discriminator" in var.name]
			gen_var_list = [var for var in tf.trainable_variables() if "generator" in var.name]

			self.dis_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.dis_loss, var_list=dis_var_list)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  			with tf.control_dependencies(update_ops):
				self.gen_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.gen_loss, var_list=gen_var_list)

			self.gen_summ = tf.summary.merge([gen_l1_summ, gen_ad_summ, trans_summ])
			self.dis_summ = tf.summary.merge([dis_loss_summ])

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.train_writer = tf.summary.FileWriter(self.graph_path+'train/')
		self.train_writer.add_graph(self.sess.graph)
		self.val_writer = tf.summary.FileWriter(self.graph_path+'val/')
		self.val_writer.add_graph(self.sess.graph)
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		no_params, gen_params = self._debug_info()
		self.file = open(self.model_path+"/loss.txt", 'w')
		self.file.write("Number of Parameters: {}\n".format(no_params))
		self.file.write("Number of Generator Parameters: {}\n".format(gen_params))

	def train_model(self, train_imgs1, train_imgs2,val_imgs1, val_imgs2, learning_rate=1e-5, batch_size=32, epoch_size=50):
		
		print "Training Images: ", train_imgs1.shape[0] 
		print "Validation Images: ", val_imgs1.shape[0]
		# print "Training Images: ", train_imgs1[0].shape[0] 
		# print "Validation Images: ", val_imgs1[0].shape[0]
		print "Learning_rate: ", learning_rate, "Batch_size", batch_size, "Epochs", epoch_size
		raw_input("Training will start above configuration. Press Enter to Start....")
		count = 0
		min_val_loss = 100.0
		with tf.name_scope("Training") as scope:
			for epoch in range(epoch_size):
				for itr in xrange(0, train_imgs1.shape[0]-batch_size, batch_size):
					# haze_in = train_imgs1[0][itr:itr+batch_size,0]
					# clear_in = train_imgs1[0][itr:itr+batch_size,1]
					# trans_in = train_imgs1[1][itr:itr+batch_size]
						
					haze_in = train_imgs1[itr:itr+batch_size,0]
					clear_in = train_imgs1[itr:itr+batch_size,1]
					trans_in = train_imgs2[itr:itr+batch_size]

					dis_in = [self.dis_solver, self.dis_loss, self.dis_summ]
					dis_out = self.sess.run(dis_in, {self.haze_in:haze_in, self.clear_in: clear_in, self.trans_in:trans_in,self.train_phase:True})

					sess_in = [self.gen_solver, self.gen_ad_loss, self.gen_L1_loss, self.gen_summ, self.trans_loss]
					gen_out = self.sess.run(sess_in, {self.haze_in:haze_in, self.clear_in: clear_in, self.trans_in:trans_in,self.train_phase:True})
					
					if itr%5==0:
						print "Epoch:", epoch, "Iteration:", itr/batch_size, "Dis Loss:", dis_out[1], "Gen L1:", gen_out[2], \
						"Gen adver:", gen_out[1], "Trans:", gen_out[4]
						self.file.write("Epoch: {} Iteration: {} Dis Loss: {} Gen L1: {} Gen adver: {} Trans: {} \n".format(epoch, itr/batch_size, dis_out[1], gen_out[2], gen_out[1], gen_out[4]))
					 
					self.train_writer.add_summary(dis_out[2], count)
					self.train_writer.add_summary(gen_out[3], count)
					count = count +1
					
				lol=0
				val_sum=0
				for itr in xrange(0, val_imgs1.shape[0]-batch_size, batch_size):
					haze_in = val_imgs1[itr:itr+batch_size,0]
					clear_in = val_imgs1[itr:itr+batch_size,1]
					trans_in = val_imgs2[itr:itr+batch_size]

					sess_in = [self.dis_loss, self.gen_ad_loss, self.gen_L1_loss,self.dis_summ, self.gen_summ, self.gen_out, self.trans_out, self.trans_loss]
					sess_out = self.sess.run(sess_in, {self.haze_in: haze_in, 
												self.clear_in: clear_in,self.trans_in:trans_in,self.train_phase:False})
					self.val_writer.add_summary(sess_out[3], count)
					self.val_writer.add_summary(sess_out[4], count)

					print "Validation Epoch:", epoch, "Iteration:", itr/batch_size, "Dis Loss:", sess_out[0], "Gen L1:", sess_out[2], \
						"Gen adver:", sess_out[1], "Trans:", sess_out[7] 
					self.file.write("Validation Epoch: {} Iteration: {} Dis Loss: {} Gen L1: {} Gen adver: {} Trans: {} \n".format(epoch, itr/batch_size, \
						sess_out[0], sess_out[2], sess_out[1], sess_out[7]))
					lol+=1
					val_sum+=sess_out[2]
				val_loss = val_sum/lol
				print "Validation L1 Loss", val_loss
				self.file.write("Epoch: {} Validation L1 Loss: {}\n".format(epoch, val_loss))
				if (val_loss<min_val_loss or epoch==epoch_size-1):
					min_val_loss = val_loss
					self.saver.save(self.sess, self.save_path+"pix2pix", global_step=epoch)
					print "Checkpoint saved"

				idx = np.random.randint(batch_size)
				stack = np.hstack((haze_in[idx], clear_in[idx], sess_out[5][idx]))
				cv2.imwrite(self.output_path +str(epoch)+"_train_img.jpg", (stack+1)*127.5)
				stack = np.hstack((trans_in[idx],sess_out[6][idx]))
				cv2.imwrite(self.output_path +str(epoch)+"_train_trans_img.jpg", (stack+1)*127.5)
			self.file.close()

	def single_img_pass(self, input_imgs, x, y, is_train):
		out_images = []
		count = 0
		start_time = time.time()
		for i in range(len(input_imgs)):
			clear = self.sess.run(y, {x:input_imgs[i:i+1], is_train:False})
			count+=1
			if count==50:
				count=0
				print (time.time()-start_time)/50.0
				start_time = time.time()
			out_images.append((clear[0]+1)/2)
		return np.array(out_images)

	# def test(self, batch_size):
	def test(self, input_imgs, batch_size):
		self.sess=tf.Session()
		
		saver = tf.train.import_meta_graph(self.save_path+'pix2pix-56.meta')
		print self.save_path
		saver.restore(self.sess,tf.train.latest_checkpoint(self.save_path))
		print self.save_path
		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("Inputs/Haze_Image:0")
		is_train = graph.get_tensor_by_name("Inputs/is_training:0")
		# y = graph.get_tensor_by_name("Model/generator/decoder/dec1/Tanh:0")
		y = graph.get_tensor_by_name("Model/generator/conv7/Tanh:0")

		# print "Tensor Loaded"
		# for itr in xrange(0, input_imgs.shape[0], batch_size):
		# 	if itr+batch_size<=input_imgs.shape[0]:
		# 		end = itr+batch_size
		# 	else:
		# 		end = input_imgs.shape[0]
		# 	input_img = input_imgs[itr:end]
		# 	out = self.sess.run(y, {x:input_img, is_train:False})
		# 	if itr==0:
		# 		tot_clr = out
		# 	else:
		# 		tot_clr = np.concatenate((tot_clr, out))
		# print "Output Shape:", tot_clr.shape
		# batch_out = (tot_clr+1)/2

		self.single_img_pass(input_imgs, x, y, is_train)

		# img_name = glob.glob("/media/mnt/dehaze/*_resize.jpg")
		# for image in img_name:
		# 	img = cv2.imread(image)
		# 	in_img = img.reshape((1, img.shape[0],img.shape[1],img.shape[2]))
		# 	out = self.sess.run(y, {x:in_img/127.5-1, is_train:False})
		# 	cv2.imwrite(image[:-4]+"_clear.jpg", (out[0]+1)*127.5)
