import torch

class Pooling(torch.nn.Module):
	def __init__(self):
		super(Pooling, self).__init__()
	def forward(self, input):
		return torch.max(input, 2)[0].contiguous()