from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import wandb
import numpy as np
import collections 
import random
from omegaconf import OmegaConf

# from models import *
import datasets
import models
from utils import evaluate, read_unknowns, nest_dict, flatten_config
# from wandb_utils import WandbData
from helpers.load_dataset import  get_train_transform, get_filtered_dataset, get_val_transform
from datasets.wilds import wilds_eval
from datetime import datetime
import subprocess
from torch.distributed import init_process_group

def setup_ddp(rank, world_size, port="12356"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def main(local_rank, world_size, args):
    setup_ddp(local_rank, world_size, str(args.port))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)

    augmentation = 'none' if not args.data.augmentation else args.data.augmentation
    augmentation = f'{augmentation}_filtered' if args.data.filter else f'{augmentation}-unfiltered'
    ckpt_name = f'checkpoint/ckpt-{args.name}-{augmentation}-{args.model}-{args.seed}-{args.hps.lr}-{args.hps.weight_decay}'
    if args.data.num_extra != 'extra':
        ckpt_name += f'-{args.data.num_extra}' # all

    # Data
    print('==> Preparing data..')
    transform = get_train_transform(args.data.base_dataset, model=args.model, augmentation=args.data.augmentation)
    val_transform = get_val_transform(args.data.base_dataset, model=args.model)
    trainset, valset, testset = get_filtered_dataset(args, transform, val_transform)
    print(f'length of trainset: {len(trainset)}')
    print(f'length of val and testset: {len(valset)}')
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=local_rank, shuffle=False)

    trainloader = DataLoader(trainset, batch_size=args.data.batch, shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)
    valloader = DataLoader(valset, batch_size=args.data.batch, shuffle=False, num_workers=8, pin_memory=True, sampler=val_sampler)
    testloader = DataLoader(testset, batch_size=args.data.batch, shuffle=False, num_workers=8, pin_memory=True, sampler=test_sampler)

    # Model
    print('==> Building model..')
    net = getattr(models, args.model)(num_classes = len(trainset.classes))
    if args.finetune:
        print("...finetuning")
        # freeze all but last layer
        for name, param in net.named_parameters(): 
            if 'fc' not in name:
                param.requires_grad = False

    net = net.to(device)

    # Optionally convert BatchNorm layers to SyncBatchNorm for better performance with variable batch sizes
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # Define criterion and optimizer before wrapping with DDP
    print("num samples per group:", collections.Counter(trainset.groups))
    print("Weights: ", trainset.class_weights)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(trainset.class_weights).to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.hps.lr,
                        momentum=0.9, weight_decay=args.hps.weight_decay)

    if args.resume or args.eval_only:  # both default false
        net, optimizer, best_acc, start_epoch = load_checkpoint(args, net, optimizer, device)

    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if args.hps.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.hps.lr_scheduler == 'custom':
        scheduler0 = torch.optim.lr_scheduler.LinearLR(optimizer, 
                        start_factor = 0.008, # The number we multiply learning rate in the first epoch
                        total_iters = 4,) # The number of iterations that multiplicative factor reaches to 1
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                milestones=[30, 60, 80], # List of epoch indices
                                gamma =0.1) # Multiplicative factor of learning rate decay
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler0, scheduler1])
    elif args.hps.lr_scheduler == 'finetune':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=0.1)
    else:
        raise ValueError("Unknown scheduler")

    # Initialize wandb only on the main process
    if local_rank == 0:
        # if args.wandb_silent:  # default false
        #     os.environ['WANDB_SILENT']="true"
        current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        proj_name = f"{args.name}-{current_time}"
        print(f"project:{args.proj}, group: {proj_name}, config: {flatten_config(args)}")
        print(f"train_size = {len(trainset)}")

    # Training
    def train(epoch):
        train_sampler.set_epoch(epoch) # 确保每个 epoch 的数据顺序不同
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, groups) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Reduce across all processes
        train_loss_tensor = torch.tensor(train_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        avg_loss = train_loss_tensor.item() / total_tensor.item()
        acc = 100. * correct_tensor.item() / total_tensor.item()

        if local_rank == 0:
            print({'train loss': avg_loss, 'train acc': acc, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})


    def gather_tensor(data, device):
        data = data.contiguous()
        local_size = torch.tensor([data.size(0)], device=device, dtype=torch.long)
        sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        sizes = [int(size.item()) for size in sizes]

        max_size = max(sizes)
        padding = (0,) * (2 * (data.dim() - 1)) + (0, max_size - data.size(0))
        data_padded = F.pad(data, padding)
        data_gathered = [torch.zeros_like(data_padded) for _ in range(world_size)]
        dist.all_gather(data_gathered, data_padded)

        data_list = []
        for i in range(world_size):
            data_list.append(data_gathered[i][:sizes[i]])

        data_all = torch.cat(data_list)
        return data_all

    def test(epoch, loader, phase='val'):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        all_targets, all_predictions, all_groups = [], [], []
        with torch.no_grad():
            for batch_idx, (inputs, targets, groups) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
            
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)

                all_targets.append(targets)
                all_predictions.append(predicted)
                all_groups.append(groups.to(device))

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Reduce across all processes
        test_loss_tensor = torch.tensor(test_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)

        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        avg_loss = test_loss_tensor.item() / total_tensor.item()
        acc = 100. * correct_tensor.item() / total_tensor.item()

        # Gather all targets and predictions
        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)
        all_groups = torch.cat(all_groups)

        all_targets = gather_tensor(all_targets, device)
        all_predictions = gather_tensor(all_predictions, device)
        all_groups = gather_tensor(all_groups, device)

        if local_rank == 0:
            all_targets = all_targets.cpu().numpy()
            all_predictions = all_predictions.cpu().numpy()
            all_groups = all_groups.cpu().numpy()

            # get per class and per group accuracies
            acc_metric, class_balanced_acc, class_acc, group_acc = evaluate(all_predictions, all_targets, all_groups)
            metrics = {"epoch": epoch, f'{phase} loss': avg_loss, f'{phase} acc': acc, f'{phase} accuracy': acc_metric, f"{phase} class accuracy": class_acc, f"{phase} balanced accuracy": class_balanced_acc, **{f"{phase} {loader.dataset.group_names[i]} acc": group_acc[i] for i in range(len(group_acc))}}
            if 'iWildCam' in args.data.base_dataset:
                wilds_metrics, _ = wilds_eval(torch.tensor(all_predictions), torch.tensor(all_targets))
                metrics.update(wilds_metrics)

            print(metrics)
            print("group acc", group_acc)

            # Save checkpoint.
            if acc > best_acc:
                if not args.eval_only or phase == 'val': # default false
                    print('Saving best ckpt..')
                    if hasattr(net, 'module'):
                        state_dict = net.module.state_dict()
                    else:
                        state_dict = net.state_dict()
                    state = {
                        'net': state_dict,
                        'acc': acc,
                        'epoch': epoch,
                        'optim': optimizer.state_dict(),
                    }
                    
                    if not os.path.exists(ckpt_name):
                        os.makedirs(ckpt_name)

                    if args.checkpoint_name: # default false
                        torch.save(state, f'./checkpoint/{args.checkpoint_name}.pth')
                        # wandb.save(f'./checkpoint/{args.checkpoint_name}.pth')
                    else:
                        torch.save(state, f'./{ckpt_name}/best.pth')
                        # wandb.save(f'./{ckpt_name}/best.pth')
                best_acc = acc
                print("best epoch = ", epoch)
                print("best val acc = ", best_acc)
                print("best group acc = ", group_acc)
                print("best balanced acc = ", class_balanced_acc)
                print("best class acc = ", class_acc)
                for i, acc_c in enumerate(class_acc):
                    print(f"class_{i:03d}_acc = {acc_c}")
        dist.barrier()

    if args.eval_only: # default false
        test(start_epoch, trainloader, phase='train_eval')
        test(start_epoch, testloader, phase='test')
    else:
        for epoch in range(start_epoch, args.epochs): # default 100
            train_sampler.set_epoch(epoch)
            train(epoch)
            test(epoch, valloader, phase='val')
            scheduler.step()
            if epoch % 10 == 0:
                test(epoch, testloader, phase='test')
        # load the best checkpoint
        dist.barrier()
        print('==> Loading best checkpoint..')
        net.module.load_state_dict(torch.load(os.path.join(ckpt_name, 'best.pth'), map_location=device)['net'])
        test(epoch, testloader, phase='test')

    dist.barrier()
    dist.destroy_process_group()

def load_checkpoint(args, net, optimizer, device):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.checkpoint_name: # default false
        checkpoint_name = f'./checkpoint/{args.checkpoint_name}'
    else:
        ckpt_name = f'checkpoint/ckpt-{args.name}-augmentation-unfiltered-{args.model}-{args.seed}-{args.hps.lr}-{args.hps.weight_decay}'
        assert os.path.exists(ckpt_name), f'{ckpt_name} does not exist'
        checkpoint_name = os.path.join(ckpt_name, 'best.pth')
    checkpoint = torch.load(checkpoint_name, map_location=device)

    net.module.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optim'])

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint at epoch {start_epoch} from {checkpoint_name}")
    return net, optimizer, best_acc, start_epoch

if __name__ == '__main__':
    from datetime import datetime
    parser = argparse.ArgumentParser(description='Dataset Understanding')
    parser.add_argument('--config', default='configs/base.yaml', help="config file")
    parser.add_argument('--overrides', nargs='*', help="Any key=value arguments to override config values "
                                                    "(use dots for.nested=overrides)")
    flags, unknown = parser.parse_known_args()

    overrides = OmegaConf.from_cli(flags.overrides)
    cfg       = OmegaConf.load(flags.config)
    base      = OmegaConf.load('configs/base.yaml')
    dataset_base = OmegaConf.load(cfg.base_config) # configs/Cub2011/base.yaml
    args      = OmegaConf.merge(base, dataset_base, cfg, overrides)
    if len(unknown) > 0:
        print(unknown)
        config = nest_dict(read_unknowns(unknown))
        to_merge = OmegaConf.create(config)
        args = OmegaConf.merge(args, to_merge)
    args.yaml = flags.config
    print(f"cfg: {cfg}")
    print(f"args: {args}")
    print(f"time start: {datetime.now()}")

    world_size = torch.cuda.device_count()
    print(f"Number of GPUs: {world_size}")
    torch.multiprocessing.spawn(main, args=(world_size,args,), nprocs=world_size)
    print(f"time end: {datetime.now()}")