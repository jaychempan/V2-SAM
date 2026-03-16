custom_imports = dict(
    imports=['projects.v2sam.evaluation'],
    allow_failed_imports=False
)
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer
from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory
from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig
from projects.v2sam.models import SAM2TrainRunner, V2SAM
from projects.v2sam.datasets import video_lisa_collate_fn, VideoObjectRelatorDataset
from projects.v2sam.models.preprocess.image_resize import DirectResize
# from projects.v2sam.evaluation.metrics import IoUMetric

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
pretrained_pth = None

# Data
template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 12
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# tokenizer = dict(
#     type=AutoTokenizer.from_pretrained,
#     pretrained_model_name_or_path=path,
#     trust_remote_code=True,
#     padding_side='right')


extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=V2SAM,
    frozen_sam2_decoder=False,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    bs=batch_size,
)

# #######################################################################
# #                      PART 3  Dataset & Dataloader                   #
# #######################################################################

# sam2 data

# video_sam2_dataset = dict(
#     type=VideoSAM2Dataset,
#     sam2_folder='/path/SA-V/sav_train',
#     expression_file='/path/SA-V/Ref-SAV-New.json',
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=4,
#     extra_image_processor=extra_image_processor,
#     sampled_frames=5,
#     select_number=5,
# )

## EgoExo-4D-Seg

# video_objectrelator_dataset = dict(
#     type=VideoObjectRelatorDataset,
#     sam2_folder='/path/Ego-Exo4D-Seg/data_segswap',
#     expression_file='/path/Ego-Exo4D-Seg/Ego2Exo_FullTrain.json',
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     mode='short',
#     repeats=1,
#     extra_image_processor=extra_image_processor,
#     sampled_frames=1,
#     select_number=5,
# )

video_handal_dataset = dict(
    type=VideoObjectRelatorDataset,
    sam2_folder='/path/datasets/HANDAL',
    expression_file='/path/Handal/handal_train_all_instruct_correct_with_prompt.json',
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    mode='short',
    repeats=1,
    extra_image_processor=extra_image_processor,
    sampled_frames=1,
    select_number=5,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # video_sam2_dataset,
        # video_objectrelator_dataset,
        video_handal_dataset
        ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

#######################################################################
#                       PART 3.5  Val Dataloader                      #
#######################################################################

# === 验证集配置 ===
# val_video_objectrelator_dataset = dict(
#     type=VideoObjectRelatorDataset,
#     sam2_folder='/path/Ego-Exo4D-Seg/data_segswap',
#     expression_file='/path/Ego-Exo4D-Seg/ego2exo_val_framelevel.json',  # <-- 验证集json
#     # sam2_folder='/path_segswap',
#     # expression_file='/home/jiancheng_pan/projects/EgoOmni/eval/egoexo_val_framelevel_newprompt_all_instruction_with_prompt.json',  # <-- 验证集json
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     mode='short',
#     extra_image_processor=extra_image_processor,
#     select_number=None,
#     sampled_frames=None,
# )

val_video_handal_dataset = dict(
    type=VideoObjectRelatorDataset,
    sam2_folder='/path/datasets/HANDAL',
    expression_file='/path/Handal/handal_test_all_instruct_correct_with_prompt.json',
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    mode='short',
    extra_image_processor=extra_image_processor,
    select_number=None,
    sampled_frames=None,
)

val_dataset = dict(
    type=ConcatDataset,
    datasets=[
        val_video_handal_dataset,
    ]
)

val_dataloader = dict(
    batch_size=1,  # 通常验证用1更稳，或与train保持一致
    num_workers=dataloader_num_workers,
    dataset=val_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type=video_lisa_collate_fn),
)

test_dataloader = val_dataloader

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1  # 每1个epoch验证一次
)
#######################################################################
#                    PART 4.5  Validation Config                      #
#######################################################################
# 验证逻辑配置
val_cfg = dict(type='ValLoop')  # 使用 mmengine 的验证循环
test_cfg = dict(type='TestLoop')

# # === 测试评估器（可选，如果暂时不想评估，可设为None） ===
val_evaluator = dict(type='SegMetric', iou_metrics=['IoU', 'Dice'])
test_evaluator = val_evaluator
#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    # checkpoint=dict(
    #     type=CheckpointHook,
    #     save_optimizer=False,
    #     by_epoch=False,
    #     interval=save_steps,
    #     max_keep_ckpts=save_total_limit),
    checkpoint = dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=True,
        interval=1
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
