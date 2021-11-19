import tensorflow as tf
import numpy as np
from python.models.sk_ops import *

class HCS(object):
	def __init__(self, compdims):
		self.compdims = compdims
		self.Hlists = []
		self.Slist = []
		self.HCSlist = []
		self.conv_shapes = []
		self.var_num = 0

	def generate_HCSs(self,conv_weights):
		self.tot_conv = len(conv_weights)
		for conv_ind in range((self.tot_conv)):
			kernel = conv_weights[conv_ind]
			[h,w,fin,fout] = kernel.shape
			self.conv_shapes.append([h,w,fin,fout])
			
			compdim = self.compdims[conv_ind]
			HCS, S, Hlist, HSCvarnum = self.generate_HCS(kernel, compdim)
			self.var_num += HSCvarnum
			self.Hlists.append(Hlist)
			self.Slist.append(S)
			self.HCSlist.append(HCS)

	def generate_HCS(self, kernel, compdim):
		Tsizes = kernel.shape
		S = self.construct_S(kernel)
		HCS = S*kernel
		Hlist = []
		for i in range(len(compdim)):
			indim = Tsizes[i]
			outdim = compdim[i]
			H = self.generate_hash_mat(indim,outdim)
			HCS = mode_n_prod_T_4(HCS, H, i)
			
			Hlist.append(H)
		HCSvarnum = tf.size(HCS)
		return HCS, S, Hlist, HCSvarnum

	def generate_rand_signed_vec(self, indim):
		s = np.random.rand(indim)
		s[s>=0.5] = 1
		s[s<0.5] = -1
		s = tf.constant(s, dtype = tf.float32)
		return s

	def generate_hash_mat(self, indim, outdim):
		h = np.random.choice(outdim, size = indim)
		H = np.zeros((indim,outdim))
		for i in range(indim):
			H[i,h[i]] = 1
		return H

	def construct_S(self, T):
		Tsizes = T.shape
		slist = []
		for i in range(len(Tsizes)):
			indim = Tsizes[i]
			s = self.generate_rand_signed_vec(indim)
			slist.append(s)
		S = out_prod_4(slist)
		return S

	def reconstruct_kernels(self):
		hcs_kernels = []
		for conv_ind in range(len(self.conv_shapes)):
			S = self.Slist[conv_ind]
			HCS = self.HCSlist[conv_ind]
			Hlist = self.Hlists[conv_ind]
			kernel = HCS
			kernel_shape = self.conv_shapes[conv_ind]
			for i in range(len(kernel_shape)):
				H = Hlist[i]
				kernel = mode_n_prod_4(kernel,H,i)
			kernel = kernel*S
			hcs_kernels.append(kernel)
		return hcs_kernels




