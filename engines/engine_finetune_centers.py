# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
from tqdm import tqdm 

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import utils.misc as misc
import utils.meta_misc as meta_misc
import utils.lr_sched as lr_sched

def cos_sim_loss(centers):
    output = F.cosine_similarity(centers, centers)
    return output.pow(2).mean()

def coding_rate_loss(W):
    """Empirical Discriminative Loss."""
    eps = 0.01
    gam1 = 1.0
    p, m = W.shape
    I = torch.eye(p).cuda()
    scalar = p / (m * eps)
    logdet = torch.logdet(I + gam1 * scalar * W.matmul(W.T))
    return logdet / 2.



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, centers = model(samples)
            # loss = criterion(outputs, targets) + cos_sim_loss(centers)
            # centers = F.normalize(centers)
            loss = criterion(outputs, targets) + 0.01 * coding_rate_loss(centers)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def class_evaluate(data_loader, model, device):
    class_true = [0.0 for i in range(64)]
    class_all = [0.0 for i in range(64)]
    model.eval()
    for batch in tqdm(data_loader, desc="class_val", leave=False):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
        for i in target:
            class_all[i] += 1
        pred_true = target[target == output.argmax(dim=1)]
        for i in pred_true:
            class_true[i] += 1

    class_acc = [class_true[i] / class_all[i] for i in range(64)]
    return class_acc


@torch.no_grad()
def get_embedding(data_loader, model, device):
    model.eval()

    embeddings = []
    targets = []
    for batch in tqdm(data_loader, desc="get_embedding", leave=False):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        output = model.forward_features(images)
        embeddings.extend(output.cpu().numpy())
        targets.extend(target)
    return embeddings, targets


@torch.no_grad()
def meta_evaluate(args, data_loader, model, device):
    model.eval()
    averager = meta_misc.Averager()
    for data, _ in tqdm(data_loader, desc="meta_val", leave=False):
        x_shot, x_query = meta_misc.split_shot_query(
            data.to(device), args.n_way, args.n_shot, args.n_query
        )
        label = meta_misc.make_query_label(args.n_way, args.n_query).to(device)

        with torch.cuda.amp.autocast():
            logits = model(x_shot, x_query).view(-1, args.n_way)

        acc = meta_misc.compute_acc(logits, label)
        averager.add(acc)
    return averager.item()
