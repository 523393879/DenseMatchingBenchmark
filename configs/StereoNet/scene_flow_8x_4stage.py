# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# This is an reimplementation of the version written by Xuanyi Li (xuanyili.edu@gmail.com)
# Original code: https://github.com/meteorshowers/StereoNet-ActiveStereoNet
# ------------------------------------------------------------------------------
import os.path as osp

# model settings
max_disp = 192
model = dict(
    meta_architecture="GeneralizedStereoModel",
    # max disparity
    max_disp=max_disp,
    # the model whether or not to use BatchNorm
    batch_norm=True,
    backbone=dict(
        type="StereoNet",
        # the in planes of feature extraction backbone
        in_planes=3,
        # the number of down-sample module used, each decrease resolution to 1/2
        downsample_num=3,
        # the number of residual block used for feature extraction
        residual_num=6,
    ),
    cost_processor=dict(
        # Use the difference between left and right feature to form cost volume, then aggregation
        type='DIF',
        cost_computation = dict(
            # default cat_fms
            type="default",
            # the maximum disparity of disparity search range under the resolution of feature
            max_disp = int(max_disp // 8),
            # the start disparity of disparity search range
            start_disp = 0,
            # the step between near disparity sample
            dilation = 1,
        ),
        cost_aggregator=dict(
            type="STEREONET",
            # the maximum disparity of disparity search range
            max_disp = max_disp,
            # the in planes of cost aggregation sub network
            in_planes=32,
        ),
    ),
    disp_predictor=dict(
        # default FasterSoftArgmin
        type='FASTER',
        # the maximum disparity of disparity search range
        max_disp = max_disp//8,
        # the start disparity of disparity search range
        start_disp = 0,
        # the step between near disparity sample
        dilation = 1,
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,

    ),
    disp_refinement=dict(
        type='STEREONET',
        # the in planes of disparity refinement sub network
        in_planes=4,
        # the number of edge aware refinement module
        num=3,
    ),
    losses=dict(
        l1_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weights for different scale loss
            weights=(1.0, 1.0, 1.0, 1.0),
            # weight for l1 loss with regard to other loss type
            weight=1.0,
        ),
    ),
    eval=dict(
        # evaluate the disparity map within (lower_bound, upper_bound)
        lower_bound=0,
        upper_bound=max_disp,
        # evaluate the disparity map in occlusion area and not occlusion
        eval_occlusion=True,
        # return the cost volume after regularization for visualization
        is_cost_return=False,
        # whether move the cost volume from cuda to cpu
        is_cost_to_cpu=True,
    ),
)

# dataset settings
dataset_type = 'SceneFlow'
# data_root = 'datasets/{}/'.format(dataset_type)
# annfile_root = osp.join(data_root, 'annotations')

root = '/home/youmin/'
# root = '/node01/jobs/io/out/youmin/'

data_root = osp.join(root, 'data/StereoMatching/', dataset_type)
annfile_root = osp.join(root, 'data/annotations/', dataset_type)

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
vis_data_root = osp.join(root, 'data/visualization_data/', dataset_type)
vis_annfile_root = osp.join(vis_data_root, 'annotations')


data = dict(
    # whether disparity of datasets is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    sparse=False,
    imgs_per_gpu=1,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_train.json'),
        input_shape=[512, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_right_disp=False,
    ),
    eval=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[544, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_right_disp=False,
    ),
    # If you don't want to visualize the results, just uncomment the vis data
    vis=dict(
        type=dataset_type,
        data_root=vis_data_root,
        annfile=osp.join(vis_annfile_root, 'vis_test.json'),
        input_shape=[544, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[544, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_right_disp=False,
    ),
)

optimizer = dict(type='RMSprop', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[15]
)
checkpoint_config = dict(
    interval=1
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# https://nvidia.github.io/apex/amp.html
apex = dict(
    # whether to use apex.synced_bn
    synced_bn=True,
    # whether to use apex for mixed precision training
    use_mixed_precision=False,
    # the model weight type: float16 or float32
    type="float16",
    # the factor when apex scales the loss value
    loss_scale=16,
)

total_epochs = 15
gpus = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = osp.join(root, 'exps/StereoNet/scene_flow_8x_4stage')

# For test
checkpoint = osp.join(work_dir, 'epoch_10.pth')
out_dir = osp.join(work_dir, 'epoch_10')
