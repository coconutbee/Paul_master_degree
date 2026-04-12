import torch


configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        MODEL_ROOT = './checkpoints', # the root to buffer your checkpoints
        LOG_ROOT = './log', # the root to log your train/val status

        BACKBONE_NAME = 'IR_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'MagFace']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension

        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        STAGES = [35, 65, 95], # epoch stages to decay learning rate

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = False,
        GPU_ID = [0], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 6,
        wandb = False,
        TestCode = False,
),
}
