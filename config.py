class Config(object):
    backbone = 'efficientnet'#
    num_classes = 2 #
    use_smooth_label=False
    loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    input_size = 384
    train_batch_size = 16  # batch size
    val_batch_size = 12
    test_batch_size = 1
    optimizer = 'adam'#sam/adam
    lr_scheduler='exp'#cosine/exp/poly
    lr = 1e-4  # adam 0.00001
    sam_lr=1e-3
    MOMENTUM = 0.9
    device = "cuda"  # cuda  or cpu
    gpu_id = [0]
    num_workers = 8  # how many workers for loading data
    max_epoch = 21
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 50
    save_interval = 2
    tensorboard_interval=50
    min_save_epoch=1
    load_from = None
    #
    log_dir = 'log/'
    train_val_data = './data/train/'
    train_label_csv = './data/train_label.csv'
    #
    checkpoints_dir = './ckpt/'
    pre_trained = '..'
