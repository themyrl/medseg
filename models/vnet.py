from monai.networks.nets.vnet import VNet 
from nnunet.network_architecture.neural_network import SegmentationNetwork


class model(SegmentationNetwork):
	def __init__(self,spatial_dims=3, in_channels=1, num_classes=1, *args, **kwargs):
		super(model, self).__init__()
		self.network = VNet(out_channels=num_classes)

	def forward(self, x, *args, **kwargs):
		return self.network(x)







