import tensorflow as tf
import numpy as np
from python.models.sk_ops import *

class Sketch(object):
	def __init__(self, l,k):
		self.l = l
		self.k = k 
		self.U1s = []
		self.U2s = []
		self.S1s = []
		self.S2s = []
		self.conv_shapes = []

	def generate_S_U(self,conv_weights):
		self.tot_conv = len(conv_weights)
		for conv_ind in range((self.tot_conv)):
			kernel = conv_weights[conv_ind]
			[h,w,fin,fout] = kernel.shape
			self.conv_shapes.append([h,w,fin,fout])
			U1, S1 = self.generate_rand_pair(kernel,4,self.k,fout)
			U2, S2 = self.generate_rand_pair(kernel,3,self.k,fin)
			self.U1s.append(U1)
			self.S1s.append(S1)
			self.U2s.append(U2)
			self.S2s.append(S2)

	def generate_rand_pair(self, kernel, sdim, indim, outdim):
		Us = []
		Ss = []
		for i in range(self.l):
			U = np.random.rand(indim,outdim)
			U[U>=0.5] = 1
			U[U<0.5] = 0
			U = U/np.sqrt(indim)
			S = mode_n_prod_4(kernel, U, sdim)
			Us.append(U)
			Ss.append(S)
		return Us, Ss

	def generate_rand_signed_mat(self, indim, outdim):
		Us = []
		for i in range(self.l):
			U = np.random.rand(indim,outdim)
			U[U>=0.5] = 1
			U[U<0.5] = 0
			U = U/np.sqrt(indim)
			Us.append(U)
		return Us

	def reconstruct_kernels(self):
		sk_kernels = []
		for conv_ind in range(len(self.conv_shapes)):
			U1_list = self.U1s[conv_ind]
			U2_list = self.U2s[conv_ind]
			S1_list = self.S1s[conv_ind]
			S2_list = self.S2s[conv_ind]
			kernel = tf.zeros(self.conv_shapes[conv_ind])
			for i in range(self.l):
				U1 = U1_list[i]
				U2 = U2_list[i]
				S1 = S1_list[i]
				S2 = S2_list[i]
				kernel = kernel + mode_n_prod_T_4(S1,U1,4)
				kernel = kernel + mode_n_prod_T_4(S2,U2,3)
			kernel = kernel/(2*self.l)
			sk_kernels.append(kernel)
		return sk_kernels




