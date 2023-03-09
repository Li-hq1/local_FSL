# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# import timm
# assert timm.__version__ == "0.3.2" # version check

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import utils.lr_decay as lrd
import utils.misc as misc

from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from datasets.datasets import CategoriesSampler, build_finetune_dataset
import models.models_vit as models_vit
from models.meta_baseline import MetaBaseline
from models.soft_focal_loss import soft_focal_loss

from engines.engine_finetune import train_one_epoch, evaluate, meta_evaluate, class_evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--dataset_name", default="mini", type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument("--save_ckpt_freq", default=200, type=int)
    parser.add_argument("--no_save_ckpt", action="store_true")
    parser.add_argument('--pooling', default='mean', type=str)
    parser.add_argument("--nextvlad_lamb", default=4, type=int)
    parser.add_argument("--nextvlad_cluster", default=64, type=int)
    parser.add_argument("--nextvlad_groups", default=8, type=int)
    parser.add_argument('--mask_ratio', default=.0, type=float)
    parser.add_argument("--focal_gamma", default=0., type=float)
    parser.add_argument("--meta_distance", default="cos", type=str)

    # few shot parameters
    parser.add_argument("--n_way", default=5, type=int)
    parser.add_argument("--n_shot", default=1, type=int)
    parser.add_argument("--n_query", default=15, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--val', action='store_true', help='Perform validation during training')
    parser.add_argument('--meta_val', action='store_true', help='Perform few-shot validation during training')
    parser.add_argument('--meta_test', action='store_true', help='Perform testing only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    args.distributed = len(args.gpus.split(",")) > 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_finetune_dataset(is_train=True, dataset_name=args.dataset_name, split="train", args=args)
    if args.val:
        dataset_val, _ = build_finetune_dataset(is_train=False, dataset_name=args.dataset_name, split="val", args=args)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        if args.val:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    global_rank = misc.get_rank()
    if global_rank == 0 and not args.meta_test:
        log_dir = Path(args.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.val:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    # Dataset & DataLoader [meta val, meta test]
    if not args.distributed:
        if args.meta_val:
            dataset_val, _ = build_finetune_dataset(
                is_train=False,
                dataset_name=args.dataset_name,
                split="meta_val",
                args=args,
            )
            sampler_val = CategoriesSampler(
                dataset_val.label, 200, args.n_way, args.n_shot + args.n_query
            )
            data_loader_metaval = torch.utils.data.DataLoader(
                dataset_val, batch_sampler=sampler_val, num_workers=8, pin_memory=True
            )
        if args.meta_test: # meta test
            dataset_test, _ = build_finetune_dataset(
                is_train=False,
                dataset_name=args.dataset_name,
                split="meta_test",
                args=args,
            )
            sampler_test_1shot = CategoriesSampler(
                dataset_test.label, 200, args.n_way, args.n_shot + args.n_query
            )
            sampler_test_5shot = CategoriesSampler(
                dataset_test.label, 200, args.n_way, args.n_shot + 4 + args.n_query
            )
            data_loader_metatest_1shot = torch.utils.data.DataLoader(
                dataset_test,
                batch_sampler=sampler_test_1shot,
                num_workers=8,
                pin_memory=True,
            )
            data_loader_metatest_5shot = torch.utils.data.DataLoader(
                dataset_test,
                batch_sampler=sampler_test_5shot,
                num_workers=8,
                pin_memory=True,
            )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        pooling=args.pooling, 
        mask_ratio=args.mask_ratio,
        nextvlad_lamb=args.nextvlad_lamb,
        nextvlad_cluster=args.nextvlad_cluster,
        nextvlad_groups=args.nextvlad_groups,
    )

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)
    meta_model = MetaBaseline(model, method=args.meta_distance).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.focal_gamma > 0:
        criterion = soft_focal_loss(gamma=args.focal_gamma)
    elif mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # manually initialize fc layer
    # trunc_normal_(model.head.weight, std=2e-5)

    # testing
    if args.meta_test:
        acc_1shot = meta_evaluate(args, data_loader_metatest_1shot, meta_model, device)
        print(f"Accuracy of the meta test 1-shot: {acc_1shot*100:.2f}% ")
        args.n_shot = 5
        acc_5shot = meta_evaluate(args, data_loader_metatest_5shot, meta_model, device)
        print(f"Accuracy of the meta test 5-shot: {acc_5shot*100:.2f}% ")
        # return
        exit(0)
    # evaluation before training
    if args.meta_val:
        acc = meta_evaluate(args, data_loader_metaval, meta_model, device)
        print(f"Accuracy of the meta val: {acc*100:.2f}%")
    if args.val:
        class_acc = class_evaluate(data_loader_val, model, device)
        for i in class_acc:
            print(i)
    

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_acc_metaval = 0.0
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and not args.no_save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        if args.val:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.meta_val:
            acc = meta_evaluate(args, data_loader_metaval, meta_model, device)
            print(f"Accuracy of the meta val: {acc*100:.2f}%")
            if max_acc_metaval < acc:
                max_acc_metaval = acc
                save_name = "best_meta_val"
                if args.output_dir and not args.no_save_ckpt:
                    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch=save_name)
            print(f"Max meta val accuracy: {max_acc_metaval*100:.2f}%")
            log_stats["meta val"] = acc
            log_stats["max_meta_val"] = max_acc_metaval

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
