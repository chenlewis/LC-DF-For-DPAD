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
# from .FMAG import Moire_fag_tensor
from .FMAG import Moire_fag_tensor
from .wavelets import DWT_2D,IDWT_2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from pywt import dwt2, idwt2
import random
import numpy as np
# 在Trainer初始化时添加
torch.multiprocessing.set_sharing_strategy('file_system')  # 必须设置
class GaussianBlurDropout(nn.Module):
    def __init__(self):
        """
        初始化 GaussianBlurDropout 类。
        """
        super(GaussianBlurDropout, self).__init__()

        # 六种不同的高斯模糊参数配置
        self.kernel_sizes = [9, 17]  # 核大小
        self.sigmas = [1.0, 2.0, 3.0]  # 方差

        # 每种配置的选择概率（六种组合）
        self.config_probs = nn.Parameter(
            torch.ones(len(self.kernel_sizes) * len(self.sigmas)) / (len(self.kernel_sizes) * len(self.sigmas)),
            requires_grad=True)

        # 每种滤波是否应用的初始概率为 0.2
        self.apply_probs = nn.Parameter(torch.full((len(self.kernel_sizes) * len(self.sigmas),), 0.2),
                                        requires_grad=False)

    def gaussian_blur(self, img, kernel_size, sigma):
        """
        应用高斯模糊。
        """
        coords = torch.arange(kernel_size) - kernel_size // 2
        grid = coords.repeat(kernel_size, 1)
        kernel = torch.exp(-(grid ** 2 + grid.T ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.to(img.device, dtype=img.dtype)
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
        # 通过 softmax 获得每种配置的选择概率分布
        config_probs = F.softmax(self.config_probs, dim=0)

        # 随机选择模糊配置
        config_choice = torch.multinomial(config_probs, num_samples=1).item()
        kernel_size = self.kernel_sizes[config_choice // len(self.sigmas)]
        sigma = self.sigmas[config_choice % len(self.sigmas)]

        # 根据应用高斯模糊的概率决定是否进行模糊
        if torch.rand(1).item() < self.apply_probs[config_choice].item():
            return self.gaussian_blur(inputs, kernel_size, sigma)
        else:
            return inputs  # 如果不应用模糊，直接返回输入

def process_batch(batch_tensor, raw_peak,a):
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
        # augmented_img = Moire_fag_tensor(img_tensor, raw_peak,a)
        #auto
        augmented_img = Moire_fag_tensor(img_tensor,  a)
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
        self.raw_peak = nn.Parameter(torch.tensor(95.0), requires_grad=True)
        self.a = nn.Parameter(torch.tensor(0.4), requires_grad=True)  # 初始值
        self.a_range = (0.3,0.7)  # a 的范围

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
        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):

            outputs = self.model(inputs, input2)  # Default forward method
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

        # =======backward and optim step only if in training phase... =========
        self.model.module.optimize_parameters(is_train, self.cls_weights)
        # loss = self.model.module.loss_G.item()

        loss = self.model.module.loss_G
        loss = loss.mean()  # 对 batch 的损失取平均
        return loss, outputs

    def get_input(self, data,isTrain):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]

        if isTrain:

            blur_module = GaussianBlurDropout()

            inputs = blur_module(inputs)

        raw_peak_clamped = torch.clamp(self.raw_peak, min=0, max=224)
        a_clamped = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])

        # 调用处理批次的函数，并传入可调整的 raw_peak
        input2 = process_batch(inputs, raw_peak=int(raw_peak_clamped.item()),a=a_clamped)

        input2_w = None
        inputs_w = None

        self.model.module.set_input(inputs, labels, input2)

        imgPath = data["imagePath"]
        return inputs, labels,imgPath,input2,inputs_w,input2_w

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

        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.lr_scheduler.get_lr
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
                    data_time.update(time.time() - end)


                    train_loss, _ = self.forward_one_batch(X, targets,input2, True)
                    self.model.train()
                    self.optimizer.zero_grad()
                    train_loss.backward()  # 计算梯度
                    self.optimizer.step()
                    if train_loss == -1:
                        # continue
                        return None

                    losses.update(train_loss, X.shape[0])
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

                logger.info(
                    "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                        data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg))

                self.lr_scheduler.step()

            self.model.eval()
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

            if epoch > 5:
                if auc >= best_metric:
                    best_metric = auc
                    best_epoch = epoch + 1
                    logger.info(
                        f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                    patience = 0

                    torch.save(self.model.module.encoder.state_dict(),
                               os.path.join(self.cfg.OUTPUT_DIR, f"model_epoch_{epoch}.pth"))
                else:
                    patience += 1
            else:
                logger.info(
                    f'Epoch {epoch}: curr_acc: {curr_acc:.3f}')
                torch.save(self.model.module.encoder.state_dict(),
                           os.path.join(self.cfg.OUTPUT_DIR, f"model_epoch_{epoch}.pth"))

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