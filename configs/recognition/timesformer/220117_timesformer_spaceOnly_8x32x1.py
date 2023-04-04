_base_ = ['../../_base_/default_runtime.py']

num_frames = 3
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=num_frames,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='space_only',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=5, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))


# dataset settings
split = 0
dataset_type = 'RawframeDataset'
data_root = '/data/syupoh/dataset/endoscopy/videos/220204'
data_root_val = '/data/syupoh/dataset/endoscopy/videos/220204'
ann_file_train = f'/data/syupoh/dataset/endoscopy/videos/220204/annotations/220204_train_{split}.txt'
ann_file_val = f'/data/syupoh/dataset/endoscopy/videos/220204/annotations/220204_test_{split}.txt'
ann_file_test = f'/data/syupoh/dataset/endoscopy/videos/220204/annotations/220204_test_{split}.txt'

batch_size = 4

# # learning policy
# lr_config = dict(policy='step', step=[5, 10])
# total_epochs = 15

# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100


evaluation = dict(  # Config of evaluation during training
    interval=5,)  # Metrics to be performed
# runtime settings
checkpoint_config = dict(interval=5)

##################
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=num_frames,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=num_frames,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
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
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))


# runtime settings
checkpoint_config = dict(interval=20)
# work_dir = './work_dirs/timesformer_divST_8x32x1_15e_kinetics400_rgb'
