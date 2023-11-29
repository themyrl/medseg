## 3D Ultrasound medical image segmentation

### 0. Installation

```
python -m venv usenv
source usenv/bin/activate

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


pip install -r requirements.txt
```


### 1. Dataset and preparation
(The dataset is currently private. Contact us to ask for an access.)

Get the original US and CT dataset in nii.gz format. Use the following commands to convert the images and the segmentation masks from the different annotators to the right format and the right 128x128x128 size. (replace `path/to/data` by your data path)

```
python convert_dataset_v2.py -i path/to/data/US_DATA/USimg -o path/to/data/US_DATA/USimg_128_t -s 128 -m trilinear -t float32
python convert_dataset_v2.py -i path/to/data/US_DATA/CTimg -o path/to/data/US_DATA/CTimg_128_t -s 128 -m trilinear -t float32

python convert_dataset_v2.py -i path/to/data/US_DATA/USmask_mf -o path/to/data/US_DATA/USmask_mf_128 -s 128 -m trilinear -t int16
python convert_dataset_v2.py -i path/to/data/US_DATA/USmask_a2 -o path/to/data/US_DATA/USmask_a2_128 -s 128 -m trilinear -t int16
python convert_dataset_v2.py -i path/to/data/US_DATA/USmask_a3 -o path/to/data/US_DATA/USmask_a3_128 -s 128 -m trilinear -t int16
python convert_dataset_v2.py -i path/to/data/CT_DATA/CTmask_mf -o path/to/data/CT_DATA/CTmask_mf_128 -s 128 -m trilinear -t int16
python convert_dataset_v2.py -i path/to/data/CT_DATA/CTmask_a2 -o path/to/data/CT_DATA/CTmask_a2_128 -s 128 -m trilinear -t int16
python convert_dataset_v2.py -i path/to/data/CT_DATA/CTmask_a3 -o path/to/data/CT_DATA/CTmask_a3_128 -s 128 -m trilinear -t int16
```

You'll nee to modify the config file to change the path of the dataset:

In `configs/dataset/us_128_double_jz_v2.yaml` and `configs/dataset/ct_128_double_jz_v2.yaml`, change the value of `path.pth` with your US or CT data folder path.

You also leverage hydra functionalities by adding `dataset.path.pth path/to/your/CT_or_US_data` to the inference or training command.


### 2. Training

To train a model from scratch with double annotations (make sure to have the model, dataset and training config file): 
```
mainDoubleV2.py -m model=model_config_name dataset=dataset_config_name training=training_config_name dataset.cv=cv_x
```


Example - To train GLAM from scratch with double annotations on CT 128x128x128 fold 1, with the same training parameters that we used:

```
python mainDoubleV2.py -m model=glam dataset=ct_128_double_jz training=training_128_jz dataset.cv=cv1
```



### 3. Inference

Don't forget to change the path of your traing models in `configs/training/your_trainer.yaml` by modifying the value of `pth` (ex: in `configs/training/training_128_jz_v2.yaml` change `pth : /path/to/your/output/folder`).

All checkpoints, model parameters and infernce segmentation masks will be saved here following this schema :
```
cfg.training.pth/cfg.dataset.name/cfg.training.name/cfg.model.name/cfg.dataset.cv
```

X NOT AVAILABLE X Get [here]() the parameters.

To evaluate a model with double annotations (make sure to have the model, dataset and training config file): 
```
mainDoubleV2.py -m model=model_config_name dataset=dataset_config_name training=training_config_name dataset.cv=cv_x training.only_val=True

```


Example - To evaluate GLAM with double annotations on CT 128x128x128 fold 1, with the same parameters that we used (be sure to have your model parameters saved as `/path/to/your/output/folder/ct_128_double_jz_v2/training_128_jz_v2/GLAM/checkpoints/latest.pt`):
```
python mainDouble.py -m model=glam dataset=ct_128_double_jz training=training_128_jz dataset.cv=cv1 training.only_val=true

```



