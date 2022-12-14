import nibabel as nib
import numpy as np
import os
import monai.transforms as T

import torch

import sys
import tqdm

import argparse


def main(args):
	all_types = {"float16":np.float16, "float32":np.float32, "float64":np.float64, "int16":np.int16}

	path = args.in_path
	out_path = args.out_path
	size = args.size
	mode = args.mode
	_type = all_types[args.type]

	if not os.path.exists(out_path):
		os.makedirs(out_path)

	for f in tqdm.tqdm(os.listdir(path)):

		if (".nii" in f):
			x = nib.load(os.path.join(path, f)).get_fdata()
			out_f = f.replace(".nii.gz", ".npz")
			if size != None:
				x = torch.from_numpy(x)
				if mode == 'trilinear':
					x = T.Resize(size, mode=mode, align_corners=True)(x[None, ...])[0,...]
				else:
					x = T.Resize(size, mode=mode)(x[None, ...])[0,...]
				x = x.numpy()
			if args.type == "int16" and mode != "nearest":
				x = T.AsDiscrete(threshold=0.5)(x)
			x = x.astype(_type)
			np.savez(os.path.join(out_path, out_f), x)






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Datset converter")
	parser.add_argument('-i', "--in_path", help="Original image folder path", required=True)
	parser.add_argument('-o', "--out_path", help="Output image folder path", required=True)
	parser.add_argument('-s', "--size", help="Output image size", default=None, type=int)
	parser.add_argument('-m', "--mode", help="Output image interpollation mode", default='nearest', choices=['area', 'trilinear', 'nearest'])
	parser.add_argument('-t', "--type", help="Output image values type", default="float16", choices=["float16", "float32", "float64", "int16"])

	args = parser.parse_args()

	main(args)
