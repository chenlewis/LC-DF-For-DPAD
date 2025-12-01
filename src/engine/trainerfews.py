#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
# import models
# import tqdm
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage
from tqdm import tqdm
import utils
logger = logging.get_logger("visual_prompt")
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR,CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from .FMAG import Moire_fag_tensor
# from .FMAG2 import Moire_fag_tensor
from .wavelets import DWT_2D,IDWT_2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from pywt import dwt2, idwt2
import random
import numpy as np

class GaussianBlurDropout(nn.Module):
    def __init__(self, apply_blur_prob=0.5):
        """
        初始化 GaussianBlurDropout 类。
        Args:
            apply_blur_prob (float): 应用高斯模糊的概率，取值范围 [0, 1]。
        """
        super(GaussianBlurDropout, self).__init__()
        # 定义模糊核大小范围 (9, 11, 13, ..., 17) 和方差选项
        self.kernel_sizes = list(range(9, 18, 2))  # 奇数核大小
        self.sigmas = [1.0, 2.0, 3.0]  # 方差

        # 可学习的概率参数，分别表示模糊核大小和方差的选择概率
        self.kernel_probs = nn.Parameter(torch.ones(len(self.kernel_sizes)) / len(self.kernel_sizes), requires_grad=True)
        self.sigma_probs = nn.Parameter(torch.ones(len(self.sigmas)) / len(self.sigmas), requires_grad=True)

        # 应用高斯模糊的概率
        self.apply_blur_prob = nn.Parameter(torch.tensor(apply_blur_prob), requires_grad=True)

    def apply_gaussian_blur(self, img, kernel_size, sigma):
        """
        应用高斯模糊。
        Args:
            img (torch.Tensor): 输入图像 (B, C, H, W)。
            kernel_size (int): 高斯核大小。
            sigma (float): 高斯核标准差。
        Returns:
            torch.Tensor: 模糊后的图像。
        """
        # 创建高斯核
        coords = torch.arange(kernel_size) - kernel_size // 2
        grid = coords.repeat(kernel_size, 1)
        kernel = torch.exp(-(grid ** 2 + grid.T ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        kernel = kernel.to(img.device, dtype=img.dtype)

        # 应用高斯模糊
        padding = kernel_size // 2
        img = F.conv2d(
            img, kernel.expand(img.size(1), 1, kernel_size, kernel_size),
            padding=padding, groups=img.size(1)
        )
        return img

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        Returns:
            torch.Tensor: 经过增强后的输入，形状为 (B, C, H, W)。
        """
        # 通过 softmax 获得模糊核大小和方差选择的概率分布
        kernel_probs = F.softmax(self.kernel_probs, dim=0)
        sigma_probs = F.softmax(self.sigma_probs, dim=0)

        # 根据概率分布随机选择模糊核大小和方差
        kernel_choice = torch.multinomial(kernel_probs, num_samples=1).item()
        sigma_choice = torch.multinomial(sigma_probs, num_samples=1).item()
        kernel_size = self.kernel_sizes[kernel_choice]
        sigma = self.sigmas[sigma_choice]

        # 根据应用高斯模糊的概率决定是否进行模糊
        if torch.rand(1).item() < self.apply_blur_prob.item():
            return self.apply_gaussian_blur(inputs, kernel_size, sigma)
        else:
            return inputs  # 如果不应用模糊，直接返回输入


class AugmentationModule(nn.Module):
    def __init__(self):
        super(AugmentationModule, self).__init__()
        # 可学习的概率参数，表示高斯模糊、拉普拉斯滤波和 JPEG 压缩的选择概率
        self.augmentation_probs = nn.Parameter(torch.tensor([0.3, 0.3, 0.4]), requires_grad=True)  # 初始化概率

    def apply_gaussian_blur(self, img, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur to the input image.
        Args:
            img (torch.Tensor): 单通道图像 (B, C, H, W)。
            kernel_size (int): 高斯核大小。
            sigma (float): 高斯核标准差。
        Returns:
            torch.Tensor: 模糊后的图像。
        """
        # 创建高斯核
        coords = torch.arange(kernel_size) - kernel_size // 2
        grid = coords.repeat(kernel_size, 1)
        kernel = torch.exp(-(grid ** 2 + grid.T ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.to(img.device, dtype=img.dtype)

        # 应用高斯模糊
        padding = kernel_size // 2
        img = F.conv2d(img, kernel.expand(img.size(1), 1, kernel_size, kernel_size), padding=padding,
                       groups=img.size(1))
        return img

    def apply_laplacian_filter(self, img):
        """
        Apply Laplacian filter to the input image.
        Args:
            img (torch.Tensor): 单通道图像 (B, C, H, W)。
        Returns:
            torch.Tensor: 经过拉普拉斯滤波后的图像。
        """
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=img.dtype, device=img.device)
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        img = F.conv2d(img, laplacian_kernel.expand(img.size(1), 1, 3, 3), padding=1, groups=img.size(1))
        return img

    def apply_jpeg_compression(self, img, quality=50):
        """
        Simulate JPEG compression using quantization in PyTorch.
        Args:
            img (torch.Tensor): 单通道图像 (B, C, H, W)，值范围为 [0, 1]。
            quality (int): JPEG 压缩质量，取值范围为 0-100。
        Returns:
            torch.Tensor: 经过 JPEG 压缩后的图像。
        """
        quality = max(10, min(quality, 100))  # 限定质量范围
        scale = (100 - quality) / 50.0
        scale = max(0.1, scale)  # 确保缩放系数有效

        # 将图像转为频域并量化
        img_freq = torch.fft.fft2(img)  # 计算傅里叶变换
        img_freq = img_freq / scale  # 模拟量化
        img_reconstructed = torch.fft.ifft2(img_freq).real  # 逆傅里叶变换

        return torch.clamp(img_reconstructed, 0, 1)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 经过增强后的输入，形状为 (B, C, H, W)。
        """
        B, C, H, W = inputs.shape

        # 根据概率选择增强方法
        augmentation_probs = F.softmax(self.augmentation_probs, dim=0)
        augmentation_choice = torch.multinomial(augmentation_probs, num_samples=1).item()

        if augmentation_choice == 0:  # 高斯模糊
            return self.apply_gaussian_blur(inputs, kernel_size=5, sigma=1.5)
        elif augmentation_choice == 1:  # 拉普拉斯滤波
            return self.apply_laplacian_filter(inputs)
        elif augmentation_choice == 2:  # JPEG 压缩
            return self.apply_jpeg_compression(inputs, quality=40)
        else:
            return inputs  # 默认直接返回

def process_batch(batch_tensor, raw_peak):
    """
    Process a batch of image tensors with Moire effect.

    Args:
        batch_tensor (torch.Tensor): Tensor of shape (B, C, H, W), batch of images.
        raw_peak (int): Raw peak value for Moire effect.

    Returns:
        torch.Tensor: Batch of augmented image tensors.
    """
    augmented_batch = []
    for img_tensor in batch_tensor:
        augmented_img = Moire_fag_tensor(img_tensor, raw_peak)
        # augmented_img = Moire_fag_tensor(img_tensor)
        augmented_batch.append(augmented_img)

    return torch.stack(augmented_batch)  # Return as a single tensor


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        # self.model = model
        self.model = model
        # print("model_set",model)
        # self.device = device
        # self.model = DDP(model.to(device))  # 确保模型移到指定的设备并包装为 DDP
        self.device = device
        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer2 = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler2 = make_scheduler(self.optimizer2, cfg.SOLVER)

        #
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")
        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        # print("model trainer",model)
        self.optimizer = utils.make_optimizer(model.parameters(), cfg['optimizer'])
        epoch_start = 1

        max_epoch = cfg.get('epoch_max')
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, max_epoch, eta_min=1e-9)
        # 初始化 DWT 模块
        # self.wavelet_dropout = WaveletDropout(initial_p=0.2)  # 实例化 WaveletDropout
        # self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        # self.lr_scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        # self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2)
    def forward_one_batch(self, inputs, targets,input2=None, is_train=True):
    # def forward_one_batch(self, inputs, targets):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device


        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )
        if input2!=None:
            input2 = input2.to(self.device, non_blocking=True)    # (batchsize, 2048)

        # print("Inputs 数据类型:", inputs.shape)
        # print("Inputs type ",type(inputs))
        # print("Targets 数据类型:", targets.shape)
        # inputs = inputs.to(self.device)  # (batchsize, 2048)
        # targets = targets.to(self.device)  # (batchsize, )
        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            if is_train:
                if (targets == 0).any():  # Check if label 1 exists in the batch
                    outputs = self.model(inputs, input2)  # Adjust forward method to accept input2
                else:
                    outputs = self.model(inputs, None)  # Default forward method
            else:
                outputs = self.model(inputs, None)  # Default forward method
            # outputs = self.model(inputs)  # (batchsize, num_cls)
            # outputs = self.model.encoder.forward_dummy(inputs)
            # print("*****output*******",outputs)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            # if is_train:
            #     # self.model.eval()
            #     # loss = self.cls_criterion(
            #     #     outputs, targets, self.cls_weights,
            #     #     self.model, inputs
            #     # )
            #     loss = self.model.loss_G(False)
            #     # print("11", loss)
            # else:
            #     # self.model.eval()
            #     # loss = self.cls_criterion(
            #     #     outputs, targets, self.cls_weights)
            #     loss = self.model.backward_G(is_train)
            #     # print("22",loss)

            # if self.cls_criterion.is_local() and is_train:
            #     self.model.eval()
            #     loss = self.cls_criterion(
            #         outputs, targets, self.cls_weights,
            #         self.model, inputs
            #     )
            #
            # elif self.cls_criterion.is_local():
            #     return torch.tensor(1), outputs
            # else:
            #     loss = self.cls_criterion(
            #         outputs, targets, self.cls_weights)
            #     # print("22", loss)
            # if loss == float('inf'):
            #     logger.info(
            #         "encountered infinite loss, skip gradient updating for this batch!"
            #     )
            #     return -1, -1
            # elif torch.isnan(loss).any():
            #     logger.info(
            #         "encountered nan loss, skip gradient updating for this batch!"
            #     )
            #     return -1, -1

        # =======backward and optim step only if in training phase... =========
        # if is_train:
            # self.model.train()
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # self.model.backward_G(True)
        # self.model.optimize_parameters(is_train,self.cls_weights)
        # loss = self.model.loss_G.item()
        self.model.module.optimize_parameters(is_train, self.cls_weights)
        loss = self.model.module.loss_G.item()
        return loss, outputs

    def get_input(self, data,isTrain):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        # self.model.set_input(inputs, labels)
        if isTrain:
            # print("input.shape",inputs.shape)
            # wavelet_dropout = WaveletDropout()  # 初始化 WaveletDropout 模块
            # blur_module = GaussianBlurDropout()
            blur_module = GaussianBlurDropout(apply_blur_prob=0.5)
            # 初始化增强模块
            # aug_module = AugmentationModule()
            # 对输入应用增强
            # inputs_w = blur_module(inputs)
            inputs = blur_module(inputs)
            inputs_w = None
            # print("input.shape", inputs.shape)
            if (labels == 0).any():  # 检查是否存在 label 为 0 的样本
                input2 = process_batch(inputs, raw_peak=95)
                # input2_w = process_batch(inputs_w, raw_peak=95)
                input2_w = None
            else:
                print("label=1")
                input2 = None
                input2_w = None
        else:
            input2 = None
            inputs_w = None
            input2_w = None
        # input2 = process_batch(inputs, raw_peak=95)
        # input2=None
        self.model.module.set_input(inputs, labels, input2)

        imgPath = data["imagePath"]
        return inputs, labels,imgPath,input2,inputs_w,input2_w
        # return inputs, labels, imgPath, input2

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        if train_loader != None:
            self.cls_weights = train_loader.dataset.get_class_weights(
                self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        else:
            self.cls_weights = test_loader.dataset.get_class_weights(
                self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # 源代码
        # self.cls_weights = train_loader.dataset.get_class_weights(
        #     self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            # lr = self.scheduler.get_lr()[0]
            lr = self.lr_scheduler.get_lr
            # self.lr_scheduler.step()
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{total_epoch}', unit='batch') as pbar:
                # losses = utils.Averager()
                for idx, input_data in enumerate(train_loader):
                    if self.cfg.DBG and idx == 20:
                        # if debugging, only need to see the first few iterations
                        break

                    X, targets,imgPath,input2,inputs_w,input2_w = self.get_input(input_data,True)
                    # print("X type ", type(X))
                    # self.model.set_input(X, targets)
                    # logger.info(X.shape)
                    # logger.info(targets.shape)
                    # measure data loading time
                    data_time.update(time.time() - end)
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is None:
                    #         print(f"Parameter {name} did not receive gradient!")
                    #     if param.data is None:
                    #         print(f"Parameter {name} did not receive data!")
                    # for name, parms in self.model.named_parameters():
                    #     if parms.data is not None and parms.grad is not None:
                    #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                    #               torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
                    # else:
                    #     print('-->name:', name, "parms.data or parms.grad is None")

                    train_loss, _ = self.forward_one_batch(X, targets,input2, True)
                    # if self.model.training:
                    #     # print("training",self.model.training)
                    #     self.model.module.set_input(inputs_w, targets, input2_w)
                    #     train_loss2, _ = self.forward_one_batch(inputs_w, targets, input2_w, True)
                    #     # loss_forward = self.model.backward_G()
                    #     train_loss = 0.5 * train_loss + 0.5 * train_loss2
                    if train_loss == -1:
                        # continue
                        return None

                    losses.update(train_loss, X.shape[0])
                    # losses.add(train_loss.item())
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    pbar.set_postfix({
                        'loss': f'{train_loss:.4f}',
                        'batch_time': f'{batch_time.val:.4f}s',
                        'data_time': f'{data_time.val:.2e}s',
                        'eta': str(datetime.timedelta(seconds=int(
                            batch_time.val * (len(train_loader) - idx - 1)
                        )))
                    })

                    # Update the progress bar
                    pbar.update(1)
                    pbar.refresh()  # Force refresh the progress bar display
                    # log during one batch
                    if (idx + 1) % log_interval == 0:
                        seconds_per_batch = batch_time.val
                        eta = datetime.timedelta(seconds=int(
                            seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                        logger.info(
                            "\tTraining {}/{}. train loss: {:.4f},".format(
                                idx + 1,
                                total_data,
                                losses.avg
                            )
                            + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                                seconds_per_batch,
                                data_time.val,
                                str(eta),
                            )
                            + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                        )
                        # for name, param in self.model.named_parameters():
                        #     print(name, "   ", param)
                logger.info(
                    "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                        data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg))
                 # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
                # self.scheduler.step()
                self.lr_scheduler.step()
                ########
                # self.model.set_input(X, targets)
                # self.model.optimize_parameters()
                # Enable eval mode


            # for name, parms in self.model.named_parameters():
            #     if parms.data is not None and parms.grad is not None:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #               torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
                # else:
                #     print('-->name:', name, "parms.data or parms.grad is None")
            self.model.eval()
            # self.model.train()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
                auc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["rocauc"]
                eer= self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["eer"]
            except KeyError:
                print("auc key error, acc key error")
                return

                # if curr_acc > best_metric:
                #     best_metric = curr_acc
                #     best_epoch = epoch + 1
                #     logger.info(
                #         f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                #     patience = 0
                #     Checkpointer(
                #         self.model,
                #         save_dir=self.cfg.OUTPUT_DIR,
                #         save_to_disk=True
                #     ).save("best_model_"+str(best_epoch))
                # else:
                #     patience += 1
            if epoch > 5:
                if auc >= best_metric:
                    best_metric = auc
                    best_epoch = epoch + 1
                    logger.info(
                        f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                    patience = 0
                    # Checkpointer(
                    #     self.model,
                    #     save_dir=self.cfg.OUTPUT_DIR,
                    #     save_to_disk=True
                    # ).save("best_model_" + str(best_epoch))
                    # torch.save(self.model.encoder.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, f"model_epoch_{epoch}.pth"))
                    torch.save(self.model.module.encoder.state_dict(),
                               os.path.join(self.cfg.OUTPUT_DIR, f"model_epoch_{epoch}.pth"))
                else:
                    patience += 1
            else:
                # Save model during the first 5 epochs
                # Checkpointer(
                #     self.model,
                #     save_dir=self.cfg.OUTPUT_DIR,
                #     save_to_disk=True
                # ).save("epoch_" + str(epoch + 1))
                logger.info(
                    f'Epoch {epoch}: curr_acc: {curr_acc:.3f}')
                torch.save(self.model.module.encoder.state_dict(),
                           os.path.join(self.cfg.OUTPUT_DIR, f"model_epoch_{epoch}.pth"))
                # torch.save(self.model.encoder.state_dict(),
                #            os.path.join(self.cfg.OUTPUT_DIR, f"model_epoch_{epoch}.pth"))
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

            # save the last checkpoints
            if self.cfg.MODEL.SAVE_CKPT:
                Checkpointer(
                    self.model,
                    save_dir=self.cfg.OUTPUT_DIR,
                    save_to_disk=True
                ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        # losses = utils.Averager()
        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)
        # 自己写的代码
        if data_loader != None:
            self.cls_weights = data_loader.dataset.get_class_weights(
                self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        else:
            self.cls_weights = data_loader.dataset.get_class_weights(
                self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        # initialize features and target
        total_logits = []
        total_targets = []
        total_imgPaths = []

        with tqdm(total=len(data_loader), desc='Processing Batches', unit='batch') as pbar:
            for idx, input_data in enumerate(data_loader):
                end = time.time()
                # X, targets, imgPath = self.get_input(input_data)
                X, targets, imgPath, input2,inputs_w,input2_w = self.get_input(input_data,False)
                # measure data loading time
                data_time.update(time.time() - end)

                if self.cfg.DBG:
                    logger.info("during eval: {}".format(X.shape))
                input2=None
                loss, outputs = self.forward_one_batch(X, targets,input2, False)
                if loss == -1:
                    return

                # losses.add(loss)
                losses.update(loss, X.shape[0])
                # measure elapsed time
                batch_time.update(time.time() - end)
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'batch_time': f'{batch_time.val:.4f}s',
                    'data_time': f'{data_time.val:.2e}s',
                    'eta': str(datetime.timedelta(seconds=int(
                        batch_time.val * (len(data_loader) - idx - 1)
                    )))
                })

                # Update the progress bar
                pbar.update(1)

                if (idx + 1) % log_interval == 0:
                    logger.info(
                        "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(
                            idx + 1,
                            total,
                            losses.avg,
                            batch_time.val,
                            data_time.val
                        ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                    )

                # targets: List[int]
                total_targets.extend(list(targets.numpy()))
                total_logits.append(outputs)
                total_imgPaths.extend(list(imgPath))

        print("total_imgPaths", len(total_imgPaths))
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg)
        )
        # print("totallogit", total_logits)
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        # print("joint",joint_logits)
        print(len(joint_logits))
        print(len(total_targets))
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL, imgPathNameList=total_imgPaths
        )
    # def eval_classifier(self, data_loader, prefix, save=False):
    #     """evaluate classifier"""
    #     # self.model.eval()
    #     batch_time = AverageMeter('Time', ':6.3f')
    #     data_time = AverageMeter('Data', ':6.3f')
    #     # losses = AverageMeter('Loss', ':.4e')
    #     losses = utils.Averager()
    #     log_interval = self.cfg.SOLVER.LOG_EVERY_N
    #     test_name = prefix + "_" + data_loader.dataset.name
    #     total = len(data_loader)
    #     # 自己写的代码
    #     if data_loader != None:
    #         self.cls_weights = data_loader.dataset.get_class_weights(
    #             self.cfg.DATA.CLASS_WEIGHTS_TYPE)
    #     else:
    #         self.cls_weights = data_loader.dataset.get_class_weights(
    #             self.cfg.DATA.CLASS_WEIGHTS_TYPE)
    #
    #     # initialize features and target
    #     total_logits = []
    #     total_targets = []
    #     total_imgPaths = []
    #     with tqdm(total=len(data_loader), desc=f'Val/test', unit='batch') as pbar:
    #         for idx, input_data in tqdm(enumerate(data_loader)):
    #             end = time.time()
    #             X, targets,imgPath = self.get_input(input_data)
    #             print("****EVAL_input****")
    #             # self.model.set_input(X, targets)
    #             # measure data loading time
    #             data_time.update(time.time() - end)
    #
    #             if self.cfg.DBG:
    #                 logger.info("during eval: {}".format(X.shape))
    #             loss, outputs = self.forward_one_batch(X, targets, False)
    #             # outputs = self.model.encoder.forward_dummy(X)
    #             if loss == -1:
    #                 return
    #             # losses.update(loss, X.shape[0])
    #             losses.add(loss)
    #             # measure elapsed time
    #             batch_time.update(time.time() - end)
    #
    #             pbar.set_postfix({
    #                 'loss': f'{loss:.4f}',
    #                 'batch_time': f'{batch_time.val:.4f}s',
    #                 'data_time': f'{data_time.val:.2e}s',
    #                 'eta': str(datetime.timedelta(seconds=int(
    #                     batch_time.val * (len(data_loader) - idx - 1)
    #                 )))
    #             })
    #
    #             # Update the progress bar
    #             pbar.update(1)
    #             pbar.refresh()  # Force refresh the progress bar display
    #             if (idx + 1) % log_interval == 0:
    #                 logger.info(
    #                     "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
    #                         idx + 1,
    #                         total,
    #                         losses.item(),
    #                         batch_time.val,
    #                         data_time.val
    #                     ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
    #                 )
    #
    #             # targets: List[int]
    #             total_targets.extend(list(targets.numpy()))
    #             total_logits.append(outputs)
    #             total_imgPaths.extend(list(imgPath))
    #         print("total_imgPaths", len(total_imgPaths))
    #         logger.info(
    #             f"Inference ({prefix}):"
    #             + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
    #                 data_time.avg, batch_time.avg)
    #             + "average loss: {:.4f}".format(losses.item()))
    #         # if self.model.side is not None:
    #         #     logger.info(
    #         #         "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
    #         # total_testimages x num_classes
    #         joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
    #         self.evaluator.classify(
    #             joint_logits, total_targets,
    #             test_name, self.cfg.DATA.MULTILABEL,imgPathNameList=total_imgPaths
    #         )
    #
    #         # save the probs and targets
    #         if save and self.cfg.MODEL.SAVE_CKPT:
    #             out = {"targets": total_targets, "joint_logits": joint_logits}
    #             out_path = os.path.join(
    #                 self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
    #             torch.save(out, out_path)
    #             logger.info(
    #                 f"Saved logits and targets for {test_name} at {out_path}")

        #
        # # save the probs and targets
        # if save and self.cfg.MODEL.SAVE_CKPT:
        #     out = {"targets": total_targets, "joint_logits": joint_logits}
        #     out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
        #     torch.save(out, out_path)
        #     logger.info(
        #         f"Saved logits and targets for {test_name} at {out_path}"
        #     )