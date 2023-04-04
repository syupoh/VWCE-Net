# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import pdb

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger, register_module_hooks
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import  load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader

import json
# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test

def cfg_args(cfg, args, dataname, dataroot):
    setname = '' if args.setname == '' else '_{0}'.format(args.setname)
    # numframes = int(args.numframes) if int(args.numframes) > 0 else cfg.train_pipeline[0].clip_len
    # attn_type = args.attntype
    # splitnumber = args.splitnumber

    
    cfg.data_root = '{0}/{1}'.format(dataroot, dataname)
    cfg.data_root_val = '{0}/{1}'.format(dataroot, dataname)
    data_prefix = '/data/syupoh/dataset/endoscopy/videos/{dataname}'.format(dataname=dataname)
    cfg.data.test.data_prefix = data_prefix
    cfg.data.val.data_prefix = data_prefix
    cfg.data.workers_per_gpu = 0

    if not args.annfile == '':
        annfilename = '{0}_{1}.txt'.format(dataname, args.annfile)
        ann_file_test = os.path.join(
            args.root,
            dataname,
            'annotations',
            annfilename
        )
        # pdb.set_trace()
        cfg.ann_file_test = ann_file_test
        cfg.data.test.ann_file = ann_file_test
        cfg.data.val.ann_file = ann_file_test
        
        if 'offset' in ann_file_test:
            cfg.data.train.with_offset = True
            cfg.data.test.with_offset = True
            cfg.data.val.with_offset = True
    else:
        cfg.ann_file_test = '{root}/{dataname}/annotations/{dataname}_test_0{setname}.txt'.format(
            root=dataroot, dataname=dataname, setname=setname)
        cfg.data.test.ann_file = cfg.ann_file_test
        cfg.data.val.ann_file = cfg.ann_file_test

    
    cfg.data.train.start_index = 0
    cfg.data.test.start_index = 0
    cfg.data.val.start_index = 0
    
    cfg.data.train.pipeline[0].frame_interval = 1
    cfg.data.test.pipeline[0].frame_interval = 1
    cfg.data.val.pipeline[0].frame_interval = 1
    
        
        
    # offset = '_offset_{0}'.format(numframes) if args.offset else ''
    # cfg.ann_file_train = '{root}/{dataname}/annotations/{dataname}_train_{splitnumber}{offset}{setname}.txt'.format(
    #     root=dataroot, dataname=dataname, offset=offset, splitnumber=splitnumber, setname=setname)
    # cfg.ann_file_val = '{root}/{dataname}/annotations/{dataname}_test_{splitnumber}{offset}{setname}.txt'.format(
    #     root=dataroot, dataname=dataname, offset=offset, splitnumber=splitnumber, setname=setname)
    # cfg.ann_file_test = '{root}/{dataname}/annotations/{dataname}_test_{splitnumber}{offset}{setname}.txt'.format(
    #     root=dataroot, dataname=dataname, offset=offset, splitnumber=splitnumber, setname=setname)
   

    # cfg.data.train.ann_file = cfg.ann_file_train
    # cfg.data.test.ann_file = cfg.ann_file_test
    # cfg.data.val.ann_file = cfg.ann_file_test

    # cfg.data.train.data_prefix = cfg.data_root
    # cfg.data.test.data_prefix = cfg.data_root_val
    # cfg.data.val.data_prefix = cfg.data_root_val
    
    
    # if attn_type != '':
    #     cfg.model.backbone.attention_type = attn_type
    #     if attn_type == 'divided_space_time':
    #         cfg.optimizer.paramwise_cfg.custom_keys['.backbone.time_embed'] = dict(decay_mult=0.0)
    #     elif attn_type == 'joint_space_time':
    #         cfg.optimizer.paramwise_cfg.custom_keys['.backbone.time_embed'] = dict(decay_mult=0.0)
    #         cfg.optimizer.lr = 0.004375
    #         cfg.optimizer.momentum = 0.9
    #     elif attn_type == 'space_only':
    #         pass

    # if numframes > 0:
    #     cfg.train_pipeline[0].clip_len = numframes
    #     cfg.test_pipeline[0].clip_len = numframes
    #     cfg.val_pipeline[0].clip_len = numframes

    #     cfg.model.backbone.num_frames = numframes
    #     cfg.data.train.pipeline[0].clip_len = numframes
    #     cfg.data.test.pipeline[0].clip_len = numframes
    #     cfg.data.val.pipeline[0].clip_len = numframes

    #     cfg.data.train.pipeline[0].num_clips = 1
    #     cfg.data.test.pipeline[0].num_clips = 1
    #     cfg.data.val.pipeline[0].num_clips = 1
        
        # cfg.data.test.pipeline[0].printing = True

        
    return cfg


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(evaluate_pth, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    load_checkpoint(model, evaluate_pth, map_location='cpu')
    
    
    model = MMDataParallel(model, device_ids=[0]) # gpu on
    outputs = single_gpu_test(model, data_loader)
    
    return outputs



def main_worker(args):
    evaluate_pth = args.evaluate_pth
    config = args.config
    workdir = os.path.dirname(evaluate_pth)

    if config == '':
        for filename in os.listdir(workdir):
            if filename.endswith('.py'):
                configfilename = filename
        
        cfg = Config.fromfile(os.path.join(workdir, configfilename))
    else:   
        cfg = Config.fromfile(config)

    if not args.dataname == '':
        dataname = args.dataname
        cfg = cfg_args(cfg, args, dataname, args.root)
        
    else:
        outname = evaluate_pth.replace('.pth', '_pred.json')
    
    outname = os.path.join(
        os.path.dirname(evaluate_pth),
        '{0}_{1}'.format(
            os.path.splitext(os.path.basename(cfg.ann_file_test))[0],
            os.path.basename(evaluate_pth))
        ).replace('.pth', '_pred.json')

    print('out : ', outname)
    # pdb.set_trace()
    if not os.path.exists(outname):
        # set cudnn benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.data.test.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        distributed = False
        

        # build the dataloader
        dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                **cfg.data.get('test_dataloader', {}))
        data_loader = build_dataloader(dataset, **dataloader_setting)

        # pdb.set_trace()
        outputs = inference_pytorch(evaluate_pth, cfg, distributed, data_loader)

        out_list = []
        with open(cfg.ann_file_test, 'r') as testtxt:
            for i, line in enumerate(testtxt.readlines()):
                temp_dict = {}
                part = line.strip().split(' ')

                # os.path.join(cfg.data_root_val, part[0])
                temp_dict['frame_dir'] = os.path.join(cfg.data_root_val, part[0])
                temp_dict['total_frames'] = int(part[1])
                if 'offset' in cfg.ann_file_test:
                    labelindex = 3
                else:
                    labelindex = 2
                temp_dict['label'] = int(part[labelindex])
                temp_dict['pred'] = outputs[i].tolist()

                out_list.append(temp_dict)

        print()
        # pdb.set_trace()
        with open(outname, 'w') as outfile:
            json.dump(out_list, outfile)
    else:
        print('Exsist!', outname)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a recognizer')
    parser.add_argument(
        '--root', 
        default='/data/syupoh/dataset/endoscopy/videos/',
    )
    parser.add_argument(
        '--config', 
        default='',
        help='train config file path'
    )
    parser.add_argument(
        '--dataname', 
        default='',
        help='train config file path'
    )
    parser.add_argument(
        '--evaluate_pth',
        type=str,
        default=''
    )
    parser.add_argument(
        '--annfile',
        type=str,
        default=''
    )
    parser.add_argument(
        '--setname', 
        default='',
        help='train config file path'
    )
    # parser.add_argument(
    #     '--numframes', 
    #     default=0,
    # )
    # parser.add_argument(
    #     '--offset', 
    #     action='store_true',
    #     # offset number is number of frames of dataset
    # )
    # parser.add_argument(
    #     '--attntype', 
    #     default='',
    # )
    # parser.add_argument(
    #     '--splitnumber', 
    #     default=0,
    # )
    # ###########################
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     help='evaluation metrics, which depends on the dataset, e.g.,'
    #     ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    # parser.add_argument(
    #     '--out',
    #     default=None,
    #     help='output result file in pkl/yaml/json format')
    # parser.add_argument(
    #     '--fuse-conv-bn',
    #     action='store_true',
    #     help='Whether to fuse conv and bn, this will slightly increase'
    #     'the inference speed')
    # parser.add_argument(
    #     '--average-clips',
    #     choices=['score', 'prob', None],
    #     default=None,
    #     help='average type when averaging test clips')
    # parser.add_argument(
    #     '--eval-options',
    #     nargs='+',
    #     action=DictAction,
    #     default={},
    #     help='custom options for evaluation, the key-value pair in xxx=yyy '
    #     'format will be kwargs for dataset.evaluate() function')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     default={},
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. For example, '
    #     "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    # ###########################
    # group_gpus = parser.add_mutually_exclusive_group()
    # group_gpus.add_argument(
    #     '--gpus',
    #     type=int,
    #     help='number of gpus to use '
    #     '(only applicable to non-distributed training)')
    # group_gpus.add_argument(
    #     '--gpu-ids',
    #     type=int,
    #     nargs='+',
    #     help='ids of gpus to use '
    #     '(only applicable to non-distributed training)')
    # parser.add_argument(
    #     '--dataname', 
    #     type=str, 
    #     default='', 
    #     help='insert dataname')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    
    if args.evaluate_pth.endswith('pth'):
        main_worker(args)
    else:
        for filename in sorted(os.listdir(args.evaluate_pth)):
            workdir = args.evaluate_pth
            if filename.endswith('pth'):
                args.evaluate_pth = os.path.join(workdir, filename)
                main_worker(args)
                
                  

if __name__ == '__main__':
    main()
