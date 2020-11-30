# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import collections
import math
import random
import torch
import pdb


###########
import contextlib
import copy
import importlib.util
import os
import sys
import time
import warnings
import collections
import numpy as np

from tqdm import tqdm
from typing import Callable, List
from collections import defaultdict

import torch
import torch.nn.functional as F

from fairseq.modules import gelu, gelu_accurate
##################


from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from copy import deepcopy

from datetime import datetime
from time import time

start_h = datetime.now()
stop_h = datetime.now()


model1500args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 2048], 'encoder_self_attention_heads': [8, 8, 8, 4, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 8, 4, 4, 4], 'decoder_ende_attention_heads': [8, 8, 8, 8, 8, 8], 'decoder_arbitrary_ende_attn': [-1, 1, 1, 1, -1, -1]}}

model1000args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 3072], 'encoder_self_attention_heads': [8, 8, 8, 4, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 4, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 8, 4], 'decoder_ende_attention_heads': [8, 8, 8, 8], 'decoder_arbitrary_ende_attn': [1, 1, 1, -1]}}

model500args = {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 2048, 2048, 2048, 2048, 2048], 'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 2, 'decoder_ffn_embed_dim': [3072, 3072], 'decoder_self_attention_heads': [8, 8], 'decoder_ende_attention_heads': [8, 8], 'decoder_arbitrary_ende_attn': [-1, -1]}}

build_start = time()
build_end = time()
sample_start = time()
sample_end = time()

modelconfigs = {'500':model500args, '1000':model1000args, '1500':model1500args}

modelargs = {}

def main(args, init_distributed=False):
    utils.import_user_module(args)
    utils.handle_save_path(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    #print(f"| Configs: {args}")

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

####################


    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(f"| Model: {args.arch} \n| Criterion: {criterion.__class__.__name__}")

    

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

      

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print(f"| Training on {args.distributed_world_size} GPUs")
    print(f"| Max tokens per GPU = {args.max_tokens} and max sentences per GPU = {args.max_sentences} \n")

    

###################################
    build_end = time()
    print(f"\n\n| **Time to build {args.lat_config}ms model from SuperT weights: {build_end - build_start}**\n\n")
##################################
    
    ###setting chosen model config from argument -> default = 1000 ms
    modelargs = modelconfigs[args.lat_config]

    

    # Measure model latency, the program will exit after profiling latency
    if args.latcpu or args.latgpu:
        for dhat in range(0, 1):
            measure_latency(args, model, dummy_src_tokens, dummy_prev, modelargs)
        exit(0)


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)

def measure_latency(args, model, dummy_src_tokens, dummy_prev, modelargs):
    # latency measurement
    assert not (args.latcpu and args.latgpu)

    sample_start = time()
    model.set_sample_config(modelargs)
    #print(modelargs)

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([30])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)

    sample_end = time()
    print(f"\n\n| **Time to sample SubT from SuperT design space: {sample_end - sample_start}**\n\n")

    if args.latcpu:
        #model_test.cpu()
        model.cpu()
        print('| Measuring model latency on CPU...')
    elif args.latgpu:
        # model_test.cuda()
        src_tokens_test = src_tokens_test.cuda()
        src_lengths_test = src_lengths_test.cuda()
        prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
        src_tokens_test.get_device()
        print('| Measuring model latency on GPU...')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    # dry runs
    for _ in range(5):
########################
        #encoder_out_test = model_test.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)
        encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

    encoder_latencies = []
    print('| Measuring encoder...')
    for _ in tqdm(range(args.latiter)):
        if args.latgpu:
            start.record()
        elif args.latcpu:
            start = time.time()

        #model_test.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)
        model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

        if args.latgpu:
            end.record()
            torch.cuda.synchronize()
            encoder_latencies.append(start.elapsed_time(end))
            if not args.latsilent:
                print('| Encoder one run on GPU: ', start.elapsed_time(end))

        elif args.latcpu:
            end = time.time()
            encoder_latencies.append((end - start) * 1000)
            if not args.latsilent:
                print('| Encoder one run on CPU: ', (end - start) * 1000)

    # only use the 10% to 90% latencies to avoid outliers
    '''print(f'| Encoder latencies: {encoder_latencies}')
    encoder_latencies.sort()
    encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]
    print(f'| Encoder latency: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')'''
    
    print(f'| Encoder latency: {encoder_latencies[0]} ms')

    # beam to the batch dimension
    # encoder_out_test_with_beam = encoder_out_test.repeat(1, args.beam)
    bsz = 1
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
    if args.latgpu:
        new_order = new_order.cuda()

    encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

    # dry runs
    for _ in range(5):
###############
        model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                           encoder_out=encoder_out_test_with_beam)

    # decoder is more complicated because we need to deal with incremental states and auto regressive things
    decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        decoder_iterations = decoder_iterations_dict['iwslt']
    elif 'wmt' in args.arch:
        decoder_iterations = decoder_iterations_dict['wmt']

    decoder_latencies = []
    print('| Measuring decoder...')
    for _ in tqdm(range(args.latiter)):
        if args.latgpu:
            start.record()
        elif args.latcpu:
            start = time.time()
        incre_states = {}
        for k_regressive in range(decoder_iterations):
########################
            model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                               encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
        if args.latgpu:
            end.record()
            torch.cuda.synchronize()
            decoder_latencies.append(start.elapsed_time(end))
            if not args.latsilent:
                print('| Decoder one run on GPU: ', start.elapsed_time(end))

        elif args.latcpu:
            end = time.time()
            decoder_latencies.append((end - start) * 1000)
            if not args.latsilent:
                print('| Decoder one run on CPU: ', (end - start) * 1000)

    # only use the 10% to 90% latencies to avoid outliers
    '''decoder_latencies.sort()
    decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]

    print(f'| Decoder latencies: {decoder_latencies}')
    print(f'| Decoder latency: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms\n')'''

    print(f'| Decoder latency: {decoder_latencies[0]} ms')

    print(f"| Overall Latency: {encoder_latencies[0] + decoder_latencies[0]} ms")


def cli_main():
    
    build_start = time()
    

    parser = options.get_training_parser()
    parser.add_argument('--train-subtransformer', action='store_true', default=False, help='whether train SuperTransformer or SubTransformer')

    #set default common config file to common.yml
    parser.add_argument('--sub-configs', required=False, default = 'configs/wmt14.en-de/subtransformer/common.yml', is_config_file=True, help='when training SubTransformer, use --configs to specify architecture and --sub-configs to specify other settings')

    #set default latency config
    parser.add_argument('--lat-config', default = '1000', help = 'default config to use from model param dictionary')

    # for profiling
    parser.add_argument('--profile-flops', action='store_true', help='measure the FLOPs of a SubTransformer')

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')

    parser.add_argument('--validate-subtransformer', action='store_true', help='evaluate the SubTransformer on the validation set')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    #############
    args.latiter = 1
    #default config file - 1000ms latency constraint
    args.configs = 'configs/wmt14.en-de/subtransformer/wmt14ende_jetson@1000ms.yml'

    #print(f"| HERE: {args}")

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    if args.latgpu or args.latcpu or args.profile_flops:
        args.distributed_world_size = 1

    if args.pdb:
        pdb.set_trace()

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
