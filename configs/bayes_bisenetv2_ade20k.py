## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    model_config='bayes',
    var_step = 200,
    momentum=0.9,
    n_cats=150,
    num_aux_heads=4,
    lr_start=1e-5,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=10000,
    dataset='ADE20k',
    im_root='/home/ethan/exp_data/ade20k',
    train_im_anns='./datasets/ade20k/train.txt',
    val_im_anns='./datasets/ade20k/val.txt',
    scales=[0.5, 2.],
    cropsize=[640, 640],
    eval_crop=[640, 640],
    eval_start_shortside=640,
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=6,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)