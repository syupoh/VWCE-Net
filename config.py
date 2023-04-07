checkpoint_config = dict(interval=20)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
num_frames = 4
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',
        num_frames=4,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.0,
        transformer_layers=None,
        attention_type='space_only',
        norm_cfg=dict(type='LN', eps=1e-06)),
    cls_head=dict(type='TimeSformerHead', num_classes=5, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
split = 0
dataset_type = 'RawframeDataset'
data_root = '/data/syupoh/dataset/endoscopy/videos//220204'
data_root_val = '/data/syupoh/dataset/endoscopy/videos//220204'
ann_file_train = '/data/syupoh/dataset/endoscopy/videos//220204/annotations/220204_train_all_0.txt'
ann_file_val = '/data/syupoh/dataset/endoscopy/videos//220204/annotations/220204_test_all_0.txt'
ann_file_test = '/data/syupoh/dataset/endoscopy/videos//220204/annotations/220204_test_all_0.txt'
batch_size = 4
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=4, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=4,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/data/syupoh/dataset/endoscopy/videos//220204/annotations/220204_train_all_0.txt',
        data_prefix='/data/syupoh/dataset/endoscopy/videos//220204',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=4,
                frame_interval=32,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        filename_tmpl='img_{:05}.png'),
    val=dict(
        type='RawframeDataset',
        ann_file=
        '/data/syupoh/dataset/endoscopy/videos//220204/annotations/220204_test_all_0.txt',
        data_prefix='/data/syupoh/dataset/endoscopy/videos//220204',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=4,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        filename_tmpl='img_{:05}.png'),
    test=dict(
        type='RawframeDataset',
        ann_file=
        '/data/syupoh/dataset/endoscopy/videos//220204/annotations/220204_test_all_0.txt',
        data_prefix='/data/syupoh/dataset/endoscopy/videos//220204',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=4,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='ThreeCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        filename_tmpl='img_{:05}.png'))
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0)
        })),
    weight_decay=0.0001,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
gpu_ids = [0]
omnisource = False
module_hooks = []
work_dir = './work_dirs/220204_normal_timesformer_spaceOnly_4x32x1'
