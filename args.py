class args():

    # hyperparameters
    put_type = 'mean'
    balance = 0.01

    # training args
    epochs = 100 # "number of training epochs, default is 2"
    save_per_epoch = 1
    batch_size = 16 # "batch size for training/testing, default is 4"
    img_path = "./Sony/Sony/"
    HEIGHT = 256
    WIDTH = 256
    size_crop = 96
    stride_crop = 90
    lr = 1e-4 # "Initial learning rate, default is 0.0001"
    lr_step = 10 # Learning rate is halved in 10 epochs
    resume = "./models/ckpt_add_pretrain_cbcr_99.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    #resume = None
    resume_pre = './models/ckpt_pretrain_cbcr80.pt'
    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."
    workers = 4
    beta = 20
    # For GPU training
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    gpu = 0


    # For testing
    test_save_dir = "/media/media/a93cbdc8-0ee1-4e69-bf9c-9ecbf5625ec9/liulu/LLIE/results/exdark_people/"
    test_visible = "./test_visible.txt"
    # test_lwir = "/media/media/a93cbdc8-0ee1-4e69-bf9c-9ecbf5625ec9/liulu/LLIE/VE-LOL-L/VE-LOL-L-Cap-Full/VE-LOL-L-Cap-Low_test/"
    test_lwir = "/media/media/a93cbdc8-0ee1-4e69-bf9c-9ecbf5625ec9/liulu/LLIE/ExDark/People/"