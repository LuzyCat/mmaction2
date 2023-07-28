# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/_base_/models/ircsn_r152.py']

# model settings
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=True,
        bn_frozen=True,
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'))

# dataset settings
dataset_type = 'VideoDataset'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))
