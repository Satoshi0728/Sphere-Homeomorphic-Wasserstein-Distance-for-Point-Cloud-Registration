import torch

class MLP_Architecture(torch.nn.Module):
	def __init__(self, emb_dims=1024,):
		# emb_dims:			Embedding Dimensions for PointNet.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(MLP_Architecture, self).__init__()

		self.emb_dims = emb_dims
		self.layers = self.create_structure()

	def create_structure(self):
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 64, 1)
		self.conv3 = torch.nn.Conv1d(64, 64, 1)
		self.conv4 = torch.nn.Conv1d(64, 128, 1)
		self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
		self.relu = torch.nn.ReLU()

		layers = [self.conv1, self.relu,
				  self.conv2, self.relu,
				  self.conv3, self.relu,
				  self.conv4, self.relu,
				  self.conv5, self.relu]
		return layers

	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		input_data = input_data.permute(0, 2, 1)
		output = input_data
		for idx, layer in enumerate(self.layers):
			output = layer(output)
		return output