_base_ = [
    '../../_base_/models/tanet_r50.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    cls_head=dict(num_classes=5))
# dataset settings
split = 0
dataset_type = 'RawframeDataset'
data_root = '/data/syupoh/dataset/endoscopy/videos/220107_lv2'
data_root_val = '/data/syupoh/dataset/endoscopy/videos/220107_lv2'
ann_file_train = f'/data/syupoh/dataset/endoscopy/videos/220107_lv2/annotations/220107_lv2_train_{split}.txt'
ann_file_val = f'/data/syupoh/dataset/endoscopy/videos/220107_lv2/annotations/220107_lv2_test_{split}.txt'
ann_file_test = f'/data/syupoh/dataset/endoscopy/videos/220107_lv2/annotations/220107_lv2_test_{split}.txt'

batch_size = 8


# learning policy
lr_config = dict(policy='step', step=[50, 75, 90])
total_epochs = 100


evaluation = dict(  # Config of evaluation during training
    interval=5,)  # Metrics to be performed

##################
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DenseSampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=batch_size,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        filename_tmpl='img_{:05}.png'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        filename_tmpl='img_{:05}.png'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        filename_tmpl='img_{:05}.png'))
evaluation = dict(
    interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.01,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# runtime settings
work_dir = './work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/'
