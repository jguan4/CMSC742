import tensorflow as tf
import numpy as np
from python.models.sk_ops import *
import tensorly as tl
from tensorly.decomposition import parafac

class CP_decomp(object):
	def __init__(self, cp_factor, cp_dim):
		self.cp_factor = cp_factor
		self.cp_dim = cp_dim
		self.var_num = 0
		self.cp_kernels = []

	def start_compressing(self, conv_weights):
		self.decomp_kernels(conv_weights)

	def decomp_kernels(self, kernels):
		for conv_ind in range(len(kernels)):
			kernel_layer = kernels[conv_ind]
			self.size = len(kernel_layer.shape)
			if self.size == 3:
				[h,fin,fout] = kernel_layer.shape
			elif self.size == 4:
				[h, w, fin, fout] = kernel_layer.shape
			cp_kernel_layer = []
			if self.cp_factor:
				cp_rank = self.cp_dim
			else:
				cp_rank = fout//self.cp_dim
			for dim in range(h):
				factors = parafac(kernel_layer[dim].numpy(), rank=cp_rank)
				cp_kernel_layer.append(factors)
			if self.size == 4:
				self.var_num += h * cp_rank * (w+fin+fout)
			elif self.size == 3:
				self.var_num += h * cp_rank * (fin+fout)

			self.cp_kernels.append(cp_kernel_layer)

	def reconstruct_kernels(self):
		kernels = []
		for conv_ind in range(len(self.cp_kernels)):
			kernel_factors = self.cp_kernels[conv_ind]
			cp_kernel_layer = []
			for dim in range(len(kernel_factors)):
				factors = kernel_factors[dim]
				cp_kernel_layer.append(tl.cp_to_tensor(factors))
			kernels.append(cp_kernel_layer)
		return kernels
