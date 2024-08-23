#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


def add_gpu_params(parser: argparse.ArgumentParser):
    parser.add_argument("--platform", default='k8s', type=str, help='platform cloud')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank')
    parser.add_argument("--rank", default=0, type=int, help='rank')
    parser.add_argument("--device", default=0, type=int, help='device')
    parser.add_argument("--world_size", default=0, type=int, help='world size')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')


def distributed_opt(args, model, opt, grad_acc=1):

    if args.platform == 'azure':
        args.hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        opt = args.hvd.DistributedOptimizer(
            opt, named_parameters=model.named_parameters(), backward_passes_per_step=grad_acc
        )
    elif args.platform == 'philly' or args.platform == 'k8s' or args.platform == 'local':
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, 
            find_unused_parameters=False, broadcast_buffers=False
        )
    return model, opt


def distributed_gather(args, tensor):
    g_y = [torch.zeros_like(tensor) for _ in range(args.world_size)]
    torch.distributed.all_gather(g_y, tensor, async_op=False)
    return torch.stack(g_y)


def distributed_sync(args):
    if args.platform == 'azure':
        args.hvd.allreduce(torch.tensor(0), name='barrier')
    else:
        args.dist.barrier()


def parse_gpu(args):
    torch.manual_seed(args.random_seed)
    
    if args.platform == 'local':
        # Check if environment variables for distributed training are set
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            args.rank = local_rank
            args.device = device
            args.world_size = torch.distributed.get_world_size()
            args.dist = dist
        else:
            # Single GPU setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            args.rank = 0
            args.device = device
            args.world_size = 1
            args.dist = None
    
    # ... rest of the function remains the same ...

    print(
        'myrank:', args.rank, 
        'local_rank:', getattr(args, 'local_rank', 'N/A'), 
        'device_count:', torch.cuda.device_count(), 
        'world_size:', args.world_size
    )
    
    
def cleanup(args):
    if args.platform == 'k8s' or args.platform == 'philly':
        args.dist.destroy_process_group()
