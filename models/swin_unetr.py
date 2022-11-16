from .utils.swin_unetr import SwinUNETR
from nnunet.network_architecture.neural_network import SegmentationNetwork


class model(SegmentationNetwork):
    def __init__(self, in_channels=1, num_classes=1, img_size=[128,128,128],  feature_size=32,*args, **kwargs):
        super(model, self).__init__()
        self.network = SwinUNETR(in_channels=in_channels, 
            out_channels=num_classes, 
            img_size=img_size, 
            feature_size=feature_size,
            depths=(2, 2, 2, 2),
            norm_name='batch')

    def forward(self, x, *args, **kwargs):
        return self.network(x)

