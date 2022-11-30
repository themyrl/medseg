from torch.utils.data import Dataset
# from monai.data import Dataset
import monai
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous, LoadImage
import random
from CustomTransform import CustomRandCropByPosNegLabeld
from monai.transforms import RandCropByPosNegLabeld, Compose
# from tools import log_debug, log_info, log_start, log_end
import numpy as np

from tools import downsample_seg_for_ds_transform3
from einops import rearrange

import torch

# import time

class CustomDataset(Dataset):
	def __init__(self, data, transform=None, iterations=250, crop_size=[128,128,128], log=None, net_num_pool_op_kernel_sizes=[], type_='train', multi_anno=True, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# We use our own Custom dataset wich with we can keep track of sub volumes position.
		self.data = monai.data.Dataset(data)
		self.iterations = iterations
		self.loader = LoadImage()
		self.n_data = len(data)
		self.transform = transform
		self.log=log
		self.type=type_
		self.multi_anno = multi_anno
		# self.croper = CustomRandCropByPosNegLabeld(
		# 				            keys=["image", "label"],
		# 				            label_key="label",
		# 				            spatial_size=crop_size,
		# 				            pos=1,
		# 				            neg=1,
		# 				            num_samples=1,
		# 				            image_key="image",
		# 				            image_threshold=0,
		# 				            log=log
		# 				        )
		self.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes
		self.idx = -1

	def __len__(self):
		if self.type == 'train':
			if self.iterations == 0:
				return len(self.data)
			return self.iterations
		else:
			return len(self.data)

	def __getitem__(self, index):
		log=self.log
		if self.type == 'train':
			if self.iterations == 0:
				self.idx += 1
				i = self.idx
			else:
				i = random.randint(0,self.n_data-1)
		else:
			self.idx += 1
			i = self.idx

		data_i = {}
		# log.debug("pth", self.data[i])
		data_i["image"] = rearrange(np.load(self.data[i]["image"])['arr_0'][None, ...], 'b x y z -> b z x y')
		data_i["label"] = rearrange(np.load(self.data[i]["label"])['arr_0'][None, ...], 'b x y z -> b z x y')
		if self.multi_anno:
			data_i["label2"] = rearrange(np.load(self.data[i]["label2"])['arr_0'][None, ...], 'b x y z -> b z x y')
			data_i["label3"] = rearrange(np.load(self.data[i]["label3"])['arr_0'][None, ...], 'b x y z -> b z x y')


		if "_3_" in self.data[i]["image"].split('/')[-1]:
			data_i["id"] = [self.data[i]["image"].split('/')[-1].replace('_3_img', '_xxx')]
		else:
			data_i["id"] = [self.data[i]["image"].split('/')[-1].replace('3_img', '_xxx')]



		shape = data_i["image"].shape

		centers = [[0,0,0]]
		if not (self.type == 'test'):
			# data_i, centers = self.croper(data_i)
			# data_i = data_i[0]
			# TODO : customise RandCropByLabelClassesd to return centers
			centers = [centers[0][2]-shape[3]//2,centers[0][0]-shape[1]//2,centers[0][1]-shape[2]//2]


		if self.transform != None:
			# Apply transformations
			tmp = apply_transform(self.transform, data_i)
			if type(tmp) == type([]):
				data_i = tmp[0] if self.transform is not None else data_i
			else:
				data_i = tmp if self.transform is not None else data_i

			data_i["image"] = torch.from_numpy(data_i["image"].numpy())
		



		data_i["center"] = np.array(centers)


		
		# Do deep supervision on labels
		if self.net_num_pool_op_kernel_sizes!=[]:
			deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
	            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

			data_i["label"] = downsample_seg_for_ds_transform3(data_i["label"][None,...].numpy(), deep_supervision_scales, classes=[0,1])
			if self.multi_anno:
				data_i["label2"] = downsample_seg_for_ds_transform3(data_i["label2"][None,...].numpy(), deep_supervision_scales, classes=[0,1])
				data_i["label3"] = downsample_seg_for_ds_transform3(data_i["label3"][None,...].numpy(), deep_supervision_scales, classes=[0,1])



		return data_i

