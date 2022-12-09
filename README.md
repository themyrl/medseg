## 3D Ultrasound medical image segmentation

### 0. Installation

```
python -m venv usenv
source usenv/bin/activate

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


pip install -r requirements.txt
```


### 1. Dataset and preparation
Get the 128x128x128 US dataset in nii.gz format. We need to convert it into .npz format.
```
python convert_dataset.py /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USimg_cropped128 /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USimg_cropped128_npz
python convert_dataset.py /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USmask_cropped128 /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USmask_cropped128_npz
```
Then change the dataset path in `outputs/US_SMALL/NNUNET/CROP_SMALL_nnu_v2\:0/.hydra/config.yaml`: `dataset.path.pth: /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022`

Change also the model path: `model.pth: /scratch/lthemyr/US/model`



### 2. Training

To train GLAM from scratch with double annotations on CT 128x128x128 fold 1:

```
python mainDouble.py -m model=glam dataset=ct_128_double_jz training=training_128_jz dataset.cv=cv1
```



### 3. Inference

X NOT AVAILABLE X Get [here](https://themyr.iiens.net/unet_128.pt) the parameters. And put the .pt file into the model folder like this: `/scratch/lthemyr/US/model/CROP_SMALL_nnu_v2/checkpoint/unet_128.pt`.

To run evaluation:

```
python mainDouble.py -m model=glam dataset=ct_128_double_jz training=training_128_jz dataset.cv=cv1
 training.only_val=true

```

if you want to evaluate on an other dataset, make sure to use .npz images and overide the `dataset.path.pth` config parameter.