import tensorflow as tf
import numpy as np
from python.models.sk_ops import *

class HCS(object):
	def __init__(self, hcs_l, hcs_k, hcs_factor):
		self.hcs_l = hcs_l
		self.hcs_k = hcs_k
		self.hcs_factor = hcs_factor
		self.Hlists = []
		self.Slist = []
		self.HCSlist = []
		self.conv_shapes = []
		self.var_num = 0

	def start_compressing(self, conv_weights):
		self.generate_HCSs_ave(conv_weights)

	def generate_HCSs_ave(self,conv_weights):
		self.tot_conv = len(conv_weights)
		for conv_ind in range((self.tot_conv)):
			kernel = conv_weights[conv_ind]
			[h,w,fin,fout] = kernel.shape
			self.conv_shapes.append([h,w,fin,fout])
			
			if self.hcs_factor:
				compdim = [int(np.ceil(fin/self.hcs_k)), int(np.ceil(fout/self.hcs_k))]
			else:
				compdim = [self.hcs_k, self.hcs_k]

			HCSlists, S_lists, Hs_lists, HCSvarnum = self.generate_HCS_ave(kernel, compdim)
			self.var_num += HCSvarnum
			self.Hlists.append(Hs_lists)
			self.Slist.append(S_lists)
			self.HCSlist.append(HCSlists)

	def generate_HCS_ave(self, kernel, compdim):
		[h,w,fin,fout] = kernel.shape
		orgdim = [fin,fout]
		S_lists = []
		HCSlists = []
		Hs_lists = []
		HCSvarnum = 0

		for l in range(self.hcs_l):
			S = self.construct_S([fin,fout])
			S = self.repeat_S(S)
			S_lists.append(S)
			HCS = S*kernel

			Hlist = []
			
			for dim in range(2):
				indim = orgdim[dim]
				outdim = compdim[dim]
				H = self.generate_hash_mat(indim,outdim)
				HCS = mode_n_prod_T_4(HCS, H, dim+2)
				Hlist.append(H)

			HCSlists.append(HCS)
			HCSvarnum += tf.size(HCS)
			Hs_lists.append(Hlist)
		return HCSlists, S_lists, Hs_lists, HCSvarnum

	def repeat_S(self,S):
		Sr = tf.tile(tf.expand_dims(tf.expand_dims(S,axis=0),axis=0),[3,3,1,1])
		return Sr

	def generate_HCSs(self,conv_weights):
		self.tot_conv = len(conv_weights)
		for conv_ind in range((self.tot_conv)):
			kernel = conv_weights[conv_ind]
			[h,w,fin,fout] = kernel.shape
			self.conv_shapes.append([h,w,fin,fout])
			
			compdim = self.hcs_k
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

	def construct_S(self, Tsizes):
		slist = []
		for i in range(len(Tsizes)):
			indim = Tsizes[i]
			s = self.generate_rand_signed_vec(indim)
			slist.append(s)
		S = out_prod_n(slist)
		return S

	def reconstruct_kernels_sin(self):
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

	def reconstruct_kernels(self):
		hcs_kernels = []
		for conv_ind in range(len(self.conv_shapes)):
			S_list = self.Slist[conv_ind]
			HCS_list = self.HCSlist[conv_ind]
			H_lists = self.Hlists[conv_ind]
			kernel_shape = self.conv_shapes[conv_ind]
			kernel = tf.zeros(kernel_shape)
			for l in range(self.hcs_l):
				S = S_list[l]
				HCS = HCS_list[l]
				Hlist = H_lists[l]
				for i in range(2):
					H = Hlist[i]
					HCS = mode_n_prod_4(HCS,H,i+2)
				HCS = HCS*S
				kernel = kernel + HCS
			kernel = kernel/(2*self.hcs_l)
			hcs_kernels.append(kernel)
		return hcs_kernels




