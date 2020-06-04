import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random


CHANNELS = 32 # MUST BE AN EVEN NUMBER
N_BLOCK  = 100
N_CLASSES = 2


class FGFunction(nn.Module):
	"""Module used for F and G
	
	Archi :
	conv -> BN -> ReLu -> conv -> BN -> ReLu
	"""
	def __init__(self, channels):
		super(FGFunction, self).__init__()
		self.bn1 = nn.BatchNorm3d(channels)
		self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

		self.bn2 = nn.BatchNorm3d(channels)
		self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)), inplace=True)
		x = F.relu(self.bn2(self.conv2(x)), inplace=True)
		return x

def revBlock(channels):
	"""Make a reversible block
	
	Arguments:
		channels {[int]} -- [number of channels fixed]
	
	Returns:
		[nn.Module] -- [The reversible block]
	"""
	innerChannels = channels // 2

	fBlock = FGFunction(innerChannels)
	gBlock = FGFunction(innerChannels)

	return rv.ReversibleBlock(fBlock, gBlock)


def revSequence(channels, n_block):
	"""Make a sequence of multiple reversible block
	
	Arguments:
		channels {[int]} -- [number of channels fixed]
		n_block {[int]} -- [Number of blocks]
	
	Returns:
		[nn.Module] -- [The reversible sequence]
	"""
	sequence = []
	for i in range(n_block):
		sequence.append(revBlock(channels))

	return rv.ReversibleSequence(nn.ModuleList(sequence))

def duplicate(x, n):
	"""Duplicate a tensor x n times by concatenation
	
	Arguments:
		x {[torch.Tensor]} -- [the tensor to duplicate]
		n {[int]} -- [number of times to duplicate the tensor]
	
	Returns:
		[torch.Tensor] -- [The tensor containing the duplicated input]
	"""

	y = torch.cat(n*[x], 1)
	
	return y



class FullyReversible(nn.Module):
	def __init__(self):
		super(FullyReversible, self).__init__()

		self.sequence = revSequence(CHANNELS, N_BLOCK)
		self.linear = nn.Linear(CHANNELS, N_CLASSES)


	def forward(self, x):
		x = duplicate(x, CHANNELS)
		x = self.sequence(x)
		x = self.linear(x.permute(0,4,2,3,1))
		return x.permute(0,4,2,3,1)

		

	@staticmethod
	def apply_argmax_softmax(pred):
		pred = F.softmax(pred, dim=1)
		return pred