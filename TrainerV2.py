import os
import torch
from monai.losses import DiceLoss
# from monai.data import DataLoader
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    RandAdjustContrastd,
    NormalizeIntensityd,
    LoadImage,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CropForegroundd,
    RandFlipd,
    Activations,
    AsDiscrete,
    Resized,
    RandSpatialCropd,
    RandCropByLabelClassesd,
)
import monai.transforms as mt
# from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice, compute_hausdorff_distance, DiceMetric

from tools import create_split_v3, import_model, get_loss, poly_lr, create_path_if_not_exists, _to_one_hot, CustomDice
from CustomTransform import CustomRandScaleCropd, CustomRandCropByPosNegLabeld
from CustomDataset import CustomDataset

from nnunet.utilities.nd_softmax import softmax_helper

from tqdm import tqdm

import nibabel as nib
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import telegram_send as ts

from einops import rearrange

import gc


from tqdm import trange


class Trainer():
    def __init__(self, cfg, log, *args, **kwargs):
        # Logs
        self.log = log
        self.dbg = cfg.training.dbg
        self.writer = SummaryWriter(log_dir='tensorboard/' + cfg.dataset.name +
                                    '_' + cfg.training.name + '_' + cfg.model.name + '_' + cfg.dataset.cv)
        self.dataset_name = cfg.dataset.name
        self.training_name = cfg.training.name
        self.model_name = cfg.model.name
        self.path = create_path_if_not_exists(os.path.join(
            cfg.training.pth, cfg.dataset.name, cfg.training.name, cfg.model.name, cfg.dataset.cv))

        # Device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.gpu)
        # torch.cuda.set_device(cfg.training.gpu)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = cfg.training.use_gpu
        torch.backends.cudnn.benchmark = True


        # Hyperparameters
        log.debug("Hyperparameters")
        self.epochs = cfg.training.epochs
        self.start_epoch = 0
        self.initial_lr = cfg.training.lr
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.crop_size = cfg.training.crop_size
        self.iterations = cfg.training.iter
        self.weight_decay = cfg.training.weight_decay
        self.net_num_pool_op_kernel_sizes = cfg.model.net_num_pool_op_kernel_sizes
        self.net_conv_kernel_sizes = cfg.model.net_conv_kernel_sizes
        self.do_clip = cfg.training.do_clip
        self.do_schedul = cfg.training.do_schedul
        self._loss = cfg.training.loss

        # Dataset
        log.debug("Dataset")
        self.online_validation = cfg.training.online_validation
        self.eval_step = cfg.training.eval_step
        self.img_size = cfg.dataset.im_size

        self.seg_path = [cfg.dataset.path.seg2, cfg.dataset.path.seg3, cfg.dataset.path.seg23]
        if "ct" in cfg.dataset.name:
            self.train_split = create_split_v3(
                cfg.dataset.path.im, self.seg_path, cv=cfg.dataset.cv, log=log, data="ct")
            self.val_split = create_split_v3(
                cfg.dataset.path.im, self.seg_path, cv=cfg.dataset.cv, val=True, log=log, data="ct")
        else:
            self.train_split = create_split_v3(
                cfg.dataset.path.im, self.seg_path, cv=cfg.dataset.cv, log=log)
            self.val_split = create_split_v3(
                cfg.dataset.path.im, self.seg_path, cv=cfg.dataset.cv, val=True, log=log)




        train_transforms = None
        test_transforms = None
        val_transforms = None

        if not 'full' in cfg.training.name:

            test_transforms = Compose(
                [
                    mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
                ])

            val_transforms = Compose(
                [
                    mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
                ])

            train_transforms = Compose(
                [
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=2),
                    RandAffined(keys=["image", "label"],
                                rotate_range=(np.pi, np.pi, np.pi),
                                translate_range=(50, 50, 50),
                                padding_mode="border",
                                scale_range=(0.25, 0.25, 0.25),
                                mode=('bilinear', 'nearest', 'nearest', 'nearest'),
                                prob=1.0),

                    Resized(keys=["image", "label"], spatial_size=self.img_size, mode="trilinear"),
                    mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    mt.RandGaussianNoised(
                        keys="image", prob=0.1, mean=0.0, std=0.1),
                    mt.RandGaussianSmoothd(keys="image", sigma_x=(
                        0.5, 1), sigma_y=(0.5, 1), sigma_z=(0.5, 1), prob=0.2),
                    mt.RandAdjustContrastd(keys="image", prob=0.15),
                ]
            )
        else:
            test_transforms = Compose(
                [
                    mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ])

            val_transforms = Compose(
                [
                    RandCropByLabelClassesd(keys=["image", "label"],
                                            label_key="label",
                                            spatial_size=self.crop_size,
                                            num_classes=cfg.dataset.classes + 1,
                                            num_samples=1
                                            ),
                    mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    
                ])

            train_transforms = Compose(
                [

                    mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    RandCropByLabelClassesd(keys=["image", "label"],
                                            label_key="label",
                                            spatial_size=self.crop_size,
                                            num_classes=cfg.dataset.classes + 1,
                                            num_samples=1
                                            ),
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=2),
                    RandAffined(keys=["image", "label"],
                                rotate_range=(np.pi, np.pi, np.pi),
                                translate_range=(50, 50, 50),
                                padding_mode="border",
                                scale_range=(0.25, 0.25, 0.25),
                                mode=('bilinear', 'nearest', 'nearest', 'nearest'),
                                prob=1.0),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    mt.RandGaussianNoised(
                        keys="image", prob=0.1, mean=0.0, std=0.1),
                    mt.RandGaussianSmoothd(keys="image", sigma_x=(
                        0.5, 1), sigma_y=(0.5, 1), sigma_z=(0.5, 1), prob=0.2),
                    mt.RandAdjustContrastd(keys="image", prob=0.15),
                ]
            )


        trainData = CustomDataset(self.train_split, transform=train_transforms, iterations=self.iterations,
                                  crop_size=self.crop_size, log=log, net_num_pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes, multi_anno=False)
        testData = CustomDataset(self.val_split, transform=test_transforms,
                                 iterations=0, crop_size=self.crop_size, log=log, type_='test', multi_anno=False)
        
        if self.online_validation:
            valData = CustomDataset(self.val_split, transform=val_transforms,
                                    iterations=0, crop_size=self.crop_size, log=log, type_='val', multi_anno=False)

        self.train_loader = DataLoader(trainData, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        log.debug('train_loader', len(self.train_loader))
        self.test_loader = DataLoader(
            testData, batch_size=1, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())
        if self.online_validation:
            self.val_loader = DataLoader(
                valData, batch_size=1, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

        self.stride = cfg.training.inference.stride
        self.classes = cfg.dataset.classes
        if self._loss == "CrossDice":
            self.classes += 1

        # Models
        log.debug("Model")
        self.feature_size = cfg.model.feature_size
        self.save_path = create_path_if_not_exists(
            os.path.join(self.path, "checkpoint"))
        self.n_save = cfg.training.checkpoint.save
        self.do_load_checkpoint = cfg.training.checkpoint.load
        self.load_path = os.path.join(self.path, "checkpoint", 'best.pt')

        self.model = import_model(cfg.model.model, dataset='US', num_classes=self.classes,
                                  num_pool=len(
                                      self.net_num_pool_op_kernel_sizes),
                                  pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
                                  conv_kernel_sizes=self.net_conv_kernel_sizes,
                                  cfg=cfg.model,
                                  log=log,
                                  img_size=self.crop_size,
                                  feature_size=self.feature_size)

        if torch.cuda.is_available() and self.use_gpu:
            # torch.cuda.set_device(cfg.training.gpu)
            self.model.cuda()
        self.model.inference_apply_nonlin = softmax_helper

        self.lr = self.initial_lr

        if cfg.training.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
        elif cfg.training.optim == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), self.lr, weight_decay=self.weight_decay)

        if self._loss == "Dice":
            # self.loss = DiceLoss(softmax=True, to_onehot_y=True)
            self.loss = DiceLoss(
                to_onehot_y=False, sigmoid=True, squared_pred=True)
        elif self._loss == "CrossDice":
            self.loss = get_loss(self.net_num_pool_op_kernel_sizes)
        elif self._loss == "CustomDice":
            self.loss = CustomDice(log)

        log.debug("Loss", self._loss)
        self.infer_path = self.path

        # if not os.path.exists(self.infer_path):
        #   os.makedirs(self.infer_path)

        if self.do_load_checkpoint:
            log.debug("Checkpoint")
            self.load_checkpoint()

    def run_training(self, *args, **kwargs):
        log = self.log
        if not self.dbg:
            ts.send(messages=["Training: " + self.dataset_name +
                              '_' + self.training_name + '_' + self.model_name])

        best_metric = -1

        log.debug('run_training')

        for epoch in range(self.start_epoch, self.epochs):

            self.model.train()


            self.optimizer.param_groups[0]['lr'] = self.lr
            btc = 0
            l_train = 0
            log.debug('epoch{}'.format(epoch))

            for batch_data in tqdm(self.train_loader):
                # log.debug('batch loaded')
                btc += 1
                self.optimizer.zero_grad()

                inputs = batch_data["image"]
                labels = batch_data["label"]
                centers = batch_data["center"]

                if torch.cuda.is_available() and self.use_gpu:
                    inputs = inputs.float().cuda(0)
                    for lab in range(len(labels)):
                        labels[lab] = labels[lab].cuda(0)


                # log.debug("inputs" ,type(inputs))

                output = self.model(inputs, centers)

                del inputs
                if len(self.net_num_pool_op_kernel_sizes) == 0:
                    labels = labels.cuda(0)
                if self._loss == "Dice" and type(output) == tuple:
                    output = output[0]
                    labels = labels[0]
                # exit(0)
                gc.collect()

        
                l = self.loss(output, labels)
                l_train += l.detach().cpu().numpy()

                gc.collect()
                l.backward()
                # exit(0)

                if self.do_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()

                for out in output:
                    del out
                del output
                for lab in labels:
                    del lab
                del labels

                gc.collect()

            l_train = l_train / btc
            l_val = 0
            if self.online_validation and (epoch % self.eval_step == 0):
                self.model.eval()
                len_val = 0
                dice_metric = DiceMetric(
                    include_background=True, reduction="mean")
                post_trans = Compose(
                    [Activations(sigmoid=True), AsDiscrete(
                        threshold=True)]
                )
                with torch.no_grad():
                    for batch_data in tqdm(self.val_loader):
                        inputs = batch_data["image"]
                        labels = batch_data["label"]
                        centers = batch_data["center"]
                        # log.debug('input', inputs.shape)
                        # log.debug('labels', labels.shape)
                        if torch.cuda.is_available() and self.use_gpu:
                            inputs = inputs.float().cuda(0)
                            labels = labels.long().cuda(0)
                        output = self.model(inputs, centers)
                        if self._loss != "Dice":
                            output = output[0]
                            output = torch.argmax(output, dim=1)
                            labels = _to_one_hot(
                                labels[0, 0, ...], num_classes=self.classes)
                            output = _to_one_hot(
                                output[0, ...], num_classes=self.classes)
                            labels = rearrange(
                                labels, 'z x y c -> c z x y')[None, ...]
                            output = rearrange(
                                output, 'z x y c -> c z x y')[None, ...]
                            l = compute_meandice(
                                output, labels, ignore_empty=False)
                            l_val += np.mean(l.cpu().numpy()[0][1:])

                        else:
                            output = post_trans(output)
                            dice_metric(y_pred=output, y=labels)
                            l = dice_metric.aggregate().item()
                            l_val += l

                        len_val += 1
                    if self._loss == "Dice":
                        dice_metric.reset()
                l_val = l_val / len_val
                if l_val > best_metric:
                    best_metric = l_val
                    best_epoch = epoch
                    self.save_chekpoint(epoch, "best.pt")

                self.writer.add_scalar('Val Dice', l_val, epoch)

            saved_txt = ""
            if (epoch + 1) % self.n_save == 0:
                self.save_chekpoint(epoch)
                saved_txt = " :: Saved!"
            log.info("Epoch: {}".format(epoch), "Train Loss: {}, Val Dice: {}, lr: {}{}".format(l_train,
                                                                                                l_val,
                                                                                                self.lr,
                                                                                                saved_txt
                                                                                                ))
            self.writer.add_scalar('Loss', l_train, epoch)
            
            self.writer.add_scalar('lr', self.lr, epoch)
            if self.do_schedul:
                self.lr = poly_lr(epoch, self.epochs, self.initial_lr, 0.9)
            torch.cuda.empty_cache()

        if not self.dbg:
            ts.send(messages=["Training END: " + self.dataset_name +
                              '_' + self.training_name + '_' + self.model_name])

    def run_eval(self, do_infer=True, *args, **kwargs):
        log = self.log
        self.load_checkpoint(os.path.join(self.path, "checkpoint", 'best.pt'))
        post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=True)]
        )

        # l_val = 0
        # len_val = 0
        # tqdm_ = trange(len(self.test_loader), desc='Bar desc', leave=True)
        if do_infer:
            self.model.eval()
            for batch_data in tqdm(self.test_loader):
                # for batch_data in self.test_loader:
                inputs = batch_data["image"]
                labels = batch_data["label"]
                prediction = self.inference(inputs, labels)

                if self._loss == "Dice":
                    prediction = post_trans(prediction)[0, ...]
                else:
                    prediction = torch.argmax(prediction, dim=1)
                idx = batch_data["id"][0][0]

                name = idx.replace('xxx', 'pred')
                file = os.path.join(self.infer_path, name)

                # pred_nib = nib.Nifti1Image(prediction.numpy(), None)
                # nib.save(pred_nib, file)
                # log.debug("file", file)
                # log.debug("idx", idx)
                np.savez(file, prediction.numpy())
            # l_val = l_val/len_val

        # loader = LoadImage()
        results = {}
        mean_all = None
        N = 0

        for f in os.listdir(self.infer_path):
            if '.npz' in f:
                pred = np.load(os.path.join(self.infer_path, f))['arr_0']
                anno = np.load(os.path.join(
                    self.seg_path, f.replace('pred', 'Vol')))['arr_0']

                if self._loss != "Dice":
                    pred = convert_seg_image_to_one_hot_encoding_batched(
                        pred[None, ...], [i for i in range(self.classes)])
                    anno = convert_seg_image_to_one_hot_encoding_batched(
                        anno[None, ...], [i for i in range(self.classes)])
                elif self._loss == "Dice":
                    pred = convert_seg_image_to_one_hot_encoding_batched(
                        pred[None, ...], [i for i in range(self.classes + 1)])
                    anno = convert_seg_image_to_one_hot_encoding_batched(
                        anno[None, ...], [i for i in range(self.classes + 1)])

                pred = torch.from_numpy(pred)
                anno = torch.from_numpy(anno)

                # if self._loss != "Dice":
                #   pred = pred[:,:,0,...]

                log.debug('anno', anno.shape)
                log.debug('pred', pred.shape)
                pred = pred[:, :, 0, ...]

                # exit(0)

                dice = compute_meandice(pred, anno, ignore_empty=False)
                hd95 = compute_hausdorff_distance(pred, anno, percentile=95)

                # if self._loss == "Dice":
                #   dice = dice.numpy()
                #   hd95 = hd95.numpy()
                # else:
                dice = dice.numpy()[0]
                hd95 = hd95.numpy()[0]

                idx = f.replace('pred', 'xxx')
                log.info(idx, "Dice: {}\nHD95: {}".format(dice, hd95))

                results[idx] = {'dice': dice.tolist(), 'hd95': hd95.tolist()}

                if mean_all == None:
                    mean_all = {'dice': np.array(dice), 'hd95': np.array(hd95)}
                else:
                    mean_all['dice'] += np.array(dice)
                    mean_all['hd95'] += np.array(hd95)
                N += 1

        mean_all['dice'] = (mean_all['dice'] / N).tolist()
        mean_all['hd95'] = (mean_all['hd95'] / N).tolist()
        results['mean_all'] = mean_all

        log.info("mean all", "Dice: {}\nHD95: {}".format(
            mean_all['dice'], mean_all['hd95']))

        with open(os.path.join(self.infer_path, 'results.json'), 'w') as outfile:
            json_string = json.dumps(results)
            outfile.write(json_string)

    def save_chekpoint(self, epoch, txt='latest.pt'):
        state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        save_this = {
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict}
        torch.save(save_this, os.path.join(self.save_path, txt))

    def load_checkpoint(self, txt=None):
        if txt == None:
            txt = self.load_path

        checkpoint = torch.load(txt)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def inference(self, inputs, labels):
        log = self.log

        # D, H, W = self.img_size
        # B, C, D_crop, H_crop, W_crop = inputs.shape

        D_crop, H_crop, W_crop = self.crop_size
        B, C, D, H, W = inputs.shape

        nD, nH, nW = int(D // (D_crop * self.stride[2])), int(
            H // (H_crop * self.stride[0])), int(W // (W_crop * self.stride[1]))

        output = torch.zeros((B, self.classes, D, H, W))
        count = torch.zeros((B, self.classes, D, H, W))

        for k in range(nD):
            for i in range(nH):
                for j in range(nW):

                    idx_d = int(k * D_crop * self.stride[0])
                    idx_h = int(i * H_crop * self.stride[1])
                    idx_w = int(j * W_crop * self.stride[2])

                    if idx_d + D_crop > D:
                        idx_d = D - D_crop
                    if idx_h + H_crop > H:
                        idx_h = H - H_crop
                    if idx_w + W_crop > W:
                        idx_w = W - W_crop

                    crop = inputs[:, :, idx_d:idx_d + D_crop,
                                  idx_h:idx_h + H_crop, idx_w:idx_w + W_crop]
                    crop_labels = labels[:, :, idx_d:idx_d + D_crop,
                                         idx_h:idx_h + H_crop, idx_w:idx_w + W_crop]

                    # log.debug("labels shape", labels.shape)
                    # exit(0)
                    if crop_labels.sum() != 0:
                        centers = [[idx_d + D_crop // 2, idx_h + H_crop //
                                    2, idx_w + W_crop // 2] for i in range(B)]
                        if torch.cuda.is_available() and self.use_gpu:
                            crop = crop.float().cuda(0)

                        with torch.no_grad():
                            out_crop = self.model(crop, centers)
                        output[:, :, idx_d:idx_d + D_crop, idx_h:idx_h +
                               H_crop, idx_w:idx_w + W_crop] = out_crop[0].cpu()
                        del out_crop

                    else:
                        output[:, :, idx_d:idx_d + D_crop, idx_h:idx_h + H_crop, idx_w:idx_w +
                               W_crop] = torch.zeros((B, self.classes, D_crop, H_crop, W_crop))

                    del crop, crop_labels

                    count[:, :, idx_d:idx_d + D_crop, idx_h:idx_h +
                          H_crop, idx_w:idx_w + W_crop] += 1

                    gc.collect()

        return rearrange(output / count, 'b c z x y -> b c x y z')
