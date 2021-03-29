#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import sys
import pdb
import numpy as np

########################################################
from time import time

model1500args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 2048], 'encoder_self_attention_heads': [8, 8, 8, 4, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 8, 4, 4, 4], 'decoder_ende_attention_heads': [8, 8, 8, 8, 8, 8], 'decoder_arbitrary_ende_attn': [-1, 1, 1, 1, -1, -1]}}

model1250args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 3072], 'encoder_self_attention_heads': [8, 8, 8, 4, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [4, 8, 8, 4, 4], 'decoder_ende_attention_heads': [8, 8, 8, 8, 8], 'decoder_arbitrary_ende_attn': [-1, 1, 1, 1, -1]}}

model1000args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 3072], 'encoder_self_attention_heads': [8, 8, 8, 4, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 4, 'decoder_ffn_embed_dim': [3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 8, 4], 'decoder_ende_attention_heads': [8, 8, 8, 8], 'decoder_arbitrary_ende_attn': [1, 1, 1, -1]}}

model750args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [3072, 3072, 3072, 2048, 3072, 3072], 'encoder_self_attention_heads': [4, 8, 4, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 2, 'decoder_ffn_embed_dim': [3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 8], 'decoder_ende_attention_heads': [8, 8, 8], 'decoder_arbitrary_ende_attn': [1, 1, 1]}}

model500args = {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 2048, 2048, 2048, 2048, 2048], 'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 2, 'decoder_ffn_embed_dim': [3072, 3072], 'decoder_self_attention_heads': [8, 8], 'decoder_ende_attention_heads': [8, 8], 'decoder_arbitrary_ende_attn': [-1, -1]}}

model350args = {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [2048, 3072, 3072, 3072, 3072, 2048], 'encoder_self_attention_heads': [8, 8, 4, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 512, 'decoder_layer_num': 1, 'decoder_ffn_embed_dim': [3072], 'decoder_self_attention_heads': [8], 'decoder_ende_attention_heads': [8], 'decoder_arbitrary_ende_attn': [-1]}}

build_start = time()
build_end = time()
inference_start = time()
inference_end = time()
lat_start = time()
lat_end = time()


modelconfigs = {'350':model350args, '500':model500args, '750':model750args, '1000':model1000args, '1250':model1250args, '1500':model1500args}

modelargs = {}

#def set_args(args):
    
    
##########################################################

def main(args):


    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    #print(args.lat_config)

    use_cuda = torch.cuda.is_available() and not args.cpu
    
    # when running on CPU, use fp32 as default
    if not use_cuda:
        args.fp16 = False

    
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    

    '''# Set dictionaries - tried in loop
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary'''


    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    torch.manual_seed(args.seed)

############################
 ##LOOP TWICE
    while True:


        print ('Please enter a latency constraint (350, 500, 750, 1000, 1250, 1500):', file = sys.stderr)
        input_lat = input()

        args.lat_config = input_lat

        print ("\n\nLatency constraint:", args.lat_config)
        print ("\n")

        ###################################
        build_start = time()
        #########################################

        ###############################
        # Load dataset splits AGAIN
        task = tasks.setup_task(args)
        task.load_dataset(args.gen_subset)
        #################################

########################################################
        with open("debug_task.txt", "a") as dFile2:
            print ("Start of loop X", file=dFile2)
            print ("\n\n\n", file=dFile2)

########################################################

###################################
        # Set dictionaries
        try:
            src_dict = getattr(task, 'source_dictionary', None)
        except NotImplementedError:
            src_dict = None
        tgt_dict = task.target_dictionary
#####################################

        # Optimize ensemble for generation
        for model in models:
            if use_cuda:
                model.cuda()

            config = utils.get_subtransformer_config(args)
            
            
            
            
            model.set_sample_config(modelconfigs[args.lat_config])
            
            
            
            print(f"| Latency: {args.lat_config} ms", file = sys.stderr)
            
           
            
            print(f"| Configs: {args}", file = sys.stderr)
            
            
            ##################################edited####################################
            
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()
            print(model, file=sys.stderr)
            print(args.path, file=sys.stderr)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)


        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(args.gen_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[model.max_positions() for model in models]
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

        ###############################    
        build_end = time()
        inference_start = time()
#########################################

        # Initialize generator
        gen_timer = StopwatchMeter()
        generator = task.build_generator(args)

        num_sentences = 0
        has_target = True
        decoder_times_all = []
        input_len_all = []
        with progress_bar.build_progress_bar(args, itr) as t:
            wps_meter = TimeMeter()
            for sample in t:

                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = sample['target'][:, :args.prefix_size]

##########################################################
                print ("\n\n\n GLOBAL VARIABLES \n\n\n", file=sys.stderr)
                print (globals(), file=sys.stderr)
                print ("\n\n\n LOCAL VARIABLES \n\n\n", file=sys.stderr)
                print (locals(), file=sys.stderr)
                print ("\n\n\n", file=sys.stderr)
                with open("debug.txt", "w") as dFile:
                    print ("\n\n\n GLOBAL VARIABLES \n\n\n", file=dFile)
                    print (globals(), file = dFile)
                    print ("\n\n\n LOCAL VARIABLES \n\n\n", file=dFile)
                    print (locals(), file = dFile)
                    print ("\n\n\n", file=dFile)

##########################################################

    ########################################################
                with open("debug_task.txt", "a") as dFile2:
                    print ("Inference Step X", file=dFile2)
                    print (len(tgt_dict), file = dFile2)
                    print ("\n\n\n", file=dFile2)

    ########################################################

                gen_timer.start()
                ###
                lat_start = time()
                hypos, decoder_times = task.inference_step(generator, models, sample, prefix_tokens)
                ###
                lat_end = time()
                input_len_all.append(np.mean(sample['net_input']['src_lengths'].cpu().numpy()))

                print(decoder_times)
                decoder_times_all.append(decoder_times)
                num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                gen_timer.stop(num_generated_tokens)

                

                for i, sample_id in enumerate(sample['id'].tolist()):
                    has_target = sample['target'] is not None


                    # Remove padding
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                    

                    # Either retrieve the original sentences or regenerate them from tokens.
                    if align_dict is not None:
                        src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                        target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                    else:
                        if src_dict is not None:
                            src_str = src_dict.string(src_tokens, args.remove_bpe)
                        else:
                            src_str = ""
                        if has_target:
                            target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                    

                    if not args.quiet:
                        if src_dict is not None:
                            print('S-{}\t{}'.format(sample_id, src_str))
                        if has_target:
                            print('T-{}\t{}'.format(sample_id, target_str))

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:args.nbest]):


                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                        )

                        


                        if not args.quiet:
                            print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                            print('P-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(
                                    lambda x: '{:.4f}'.format(x),
                                    hypo['positional_scores'].tolist(),
                                ))
                            ))

                            if args.print_alignment:
                                print('A-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                                ))

                wps_meter.update(num_generated_tokens)
                t.log({'wps': round(wps_meter.avg)})
                num_sentences += sample['nsentences']


    ####################################
        inference_end = time()
        print(f"\n| **Time to set up model and sample: {build_end - build_start}**\n")
        print(f"\n| **Time for full inference: {inference_end - inference_start}**\n")
        print(f"\n| **Time for full inference: {lat_end - lat_start}**\n")
        #tgt_dict = tgt_dict_backup
        

    ########################################################

    ############################edited###########################

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--encoder-embed-dim-subtransformer', type=int, help='subtransformer encoder embedding dimension',
                        default=None)
    parser.add_argument('--decoder-embed-dim-subtransformer', type=int, help='subtransformer decoder embedding dimension',
                        default=None)

    parser.add_argument('--encoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)
    parser.add_argument('--decoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)

    parser.add_argument('--encoder-layer-num-subtransformer', type=int, help='subtransformer num encoder layers')
    parser.add_argument('--decoder-layer-num-subtransformer', type=int, help='subtransformer num decoder layers')

    parser.add_argument('--encoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
    parser.add_argument('--decoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
    parser.add_argument('--decoder-ende-attention-heads-all-subtransformer', nargs='+', default=None, type=int)

    parser.add_argument('--decoder-arbitrary-ende-attn-all-subtransformer', nargs='+', default=None, type=int)

    #set default latency config
    parser.add_argument('--lat-config', default = '1000', help = 'default config to use from model param dictionary')
    ###########################
    
    args = options.parse_args_and_arch(parser)

    if args.pdb:
        pdb.set_trace()

    main(args)


if __name__ == '__main__':
    cli_main()
