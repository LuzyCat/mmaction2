model = dict(
    backbone=dict(
        channel_ratio=8,
        fast_pathway=dict(
            base_channels=8,
            conv1_kernel=(
                5,
                7,
                7,
            ),
            conv1_stride_t=1,
            depth=50,
            lateral=False,
            norm_eval=False,
            pool1_stride_t=1,
            pretrained=None,
            type='resnet3d'),
        pretrained=None,
        resample_rate=8,
        slow_pathway=dict(
            conv1_kernel=(
                1,
                7,
                7,
            ),
            conv1_stride_t=1,
            depth=50,
            dilations=(
                1,
                1,
                1,
                1,
            ),
            inflate=(
                0,
                0,
                1,
                1,
            ),
            lateral=True,
            norm_eval=False,
            pool1_stride_t=1,
            pretrained=None,
            type='resnet3d'),
        speed_ratio=8,
        type='ResNet3dSlowFast'),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=2304,
        num_classes=55,
        spatial_type='avg',
        type='SlowFastHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')

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

