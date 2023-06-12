_base_ = [
    '/mnt/sda/2022-0526/home/scc/ytx/Software/mmagic/configs/biggan/biggan-deep_cvt-hugging-face_rgb_imagenet1k-256x256.py',
]

dataset_type = 'CubBirdDataset'

train_pipeline = [
    dict(type='LoadImageFromFile',  # Load images from files
        key='gt',  # Keys in results to find the corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='Resize', keys=['gt'], scale=(256, 256)),
    dict(type='Normalize', keys=['gt'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], to_rgb=True),
    # dict(type='ToTensor', keys=['gt']),
    # dict(type='ImageToTensor', keys=['gt']),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    _delete_ = True,
    num_workers=8, 
    batch_size=32,
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=True),  # The type of data sampler
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011',
        ann_file='squareCroppedImages_train_annotation.txt',
        data_prefix="/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011",
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_ = True,
    batch_size=128,
    num_workers=8, 
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=True),  # The type of data sampler
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011',
        ann_file='squareCroppedImages_test_annotation.txt',
        data_prefix="/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011",
        pipeline=train_pipeline))

test_dataloader = val_dataloader

train_cfg = dict(
    _delete_ = True,
    type='IterBasedTrainLoop',  # The name of train loop type
    max_iters=300000,  # The number of total iterations
    val_interval=500,  # The number of validation interval iterations
)


num_classes=120
model = dict(
    num_classes=num_classes,
    generator=dict(
        num_classes=num_classes,),
    discriminator=dict(
        num_classes=num_classes,))

# optim_wrapper = dict(
#     _delete_ = True,
#     discriminator=dict(
#         _delete_ = True,
#         type='OptimWrapper',
#         optimizer=dict(type='Adam', lr=0.00001),
#     )
# )

optim_wrapper = dict(
    _delete_ = True,
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        optimizer=dict(type='Adam', lr=0.0016, betas=(0, 0.9919919678228657))),
    discriminator=dict(
        optimizer=dict(
            type='Adam',
            lr=0.0018823529411764706,
            betas=(0, 0.9905854573074332))))

param_scheduler = dict(  # Config of learning policy
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5
)
work_dir = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/work_dirs/StanfordDogs_test"
# default_hooks = dict(visualization=dict(type='DetVisualizationHook', draw=True))
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=500,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]