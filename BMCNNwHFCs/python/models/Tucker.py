import tensorflow as tf
import numpy as np
from python.models.sk_ops import *
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor

class Tucker(object):
	def __init__(self, Tucker_factor, Tucker_rank):
		self.Tucker_factor = Tucker_factor
		self.Tucker_rank = Tucker_rank
		self.var_num = 0
		self.cores = []
		self.factor = []

	def decomp_kernels(self, kernels):
		for conv_ind in range(len(kernels)):
			kernel_layer = kernels[conv_ind]
			self.size = len(kernel_layer.shape)
			if self.size == 3:
				[h,fin,fout] = kernel_layer.shape
			elif self.size == 4:
				[h, w, fin, fout] = kernel_layer.shape
			tucker_kernel_layer = []
			if self.Tucker_factor:
				tucker_rank = [int(np.ceil(fin/self.Tucker_rank)),fout//self.Tucker_rank]
			else:
				tucker_rank = [self.Tucker_rank, self.Tucker_rank]
			if self.size == 4:
				tucker_rank = [3]+tucker_rank
			for dim in range(h):
				factors = tucker(kernel_layer[dim].numpy(), rank=tucker_rank)
				tucker_kernel_layer.append(factors)
			self.cores.append(tucker_kernel_layer)
			if self.size == 4:
				self.var_num += h*(np.sum(tucker_rank) + np.dot([w,fin,fout], tucker_rank)) 
			elif self.size == 3:
				self.var_num += h*(np.sum(tucker_rank) + np.dot([fin,fout], tucker_rank))

	def reconstruct_kernels(self):
		kernels = []
		for conv_ind in range(len(self.cores)):
			tucker_kernels = self.cores[conv_ind]
			Tucker_kernel_layer = []
			for dim in range(len(tucker_kernels)):
				factor = tucker_kernels[dim]
				Tucker_kernel_layer.append(tucker_to_tensor(factor))
			kernels.append(Tucker_kernel_layer)
		return kernels