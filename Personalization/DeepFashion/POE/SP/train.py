import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np 
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, MagFace
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy, de_preprocess
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import os
import shutil
import datetime
from util.Validation_VGGFace2 import Validation_VGGFace2
from util.Validation_IJBC_Covariate import Validation_IJBC_Covariate
from util.DataLoader import FaceIdPoseDataset_Pairs
from util.util import foolproof_dataset, verification_list
from util.triplet_loss_batch_hard import TripletLoss_BatchHard
import torchvision
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do ijb test')
    parser.add_argument('--TRAIN_DATA_ROOT', type=str, default='', help='the parent root where your train/val/test data are stored')
    parser.add_argument('--TEST_IJBC_DATA_ROOT', type=str, default='', help='the parent root where your train/val/test data are stored')
    # Pre-trained settings
    parser.add_argument('--BACKBONE_RESUME_ROOT', type=str, default='./Pretrained/ms1m_ir50/backbone_ir50_ms1m_epoch63.pth', help='choice your pt')
    parser.add_argument('--HEAD_RESUME_ROOT', type=str, default='', choices=[None])
    # Evaluation path
    parser.add_argument('--BASE_FILE_PATH', type=str, default='./DataList', choices=[None])
    parser.add_argument('--IJBA_Protocol_Path', type=str, default='./IJB-A_protocols/Train_Compare_List', help='IJB-A Protocol Path')
    parser.add_argument('--IJBC_Protocol_Path', type=str, default='./IJB-C_protocols/_Processed_Protocols', help='IJB-C Protocol Path')
    parser.add_argument('--CSV_VGGFace2_FILE', type=str, default='', help='VGGFace2 DataList (for training/test)')
    parser.add_argument('--CSV_IJBA_FILE', type=str, default='IJB-A_FOCropped_250_250_84', help='IJB-A DataList')
    parser.add_argument('--CSV_IJBC_FILE', type=str, default='IJBC_Official_Aligned', help='IJB-C DataList')
    parser.add_argument('--CSV_MPIE_FILE', type=str, default='MPIE_FOCropped_250_250_84_GalleryProbe', help='MPIE DataList')
    # Hyper-parameters
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help='batch size')
    parser.add_argument('--NUM_EPOCH', type=int, default=70, help='maximum epochs')
    parser.add_argument('--SAVE_FREQ', type=int, default=10, help='how frequen to save the model')
    parser.add_argument('--LR', type=float, default=0.01, help='learning rate')
    parser.add_argument('--WEIGHT_DECAY', type=float, default=5e-4, help='WEIGHT_DECAY')
    parser.add_argument('--MOMENTUM', type=float, default=0.9, help='MOMENTUM')
    # Losses
    parser.add_argument('--Triplet_Loss', type=bool, default=True, help='use triplet loss')
    parser.add_argument('--lambda_triplet', type=float, default=1, help='weight of triplet loss')
    parser.add_argument('--lambda_triplet_margin', type=float, default=0.35, help='margin of triplet loss (cosine dist [0, 1])')
    # Evaluation
    parser.add_argument('--Verification_Test_IJBC', type=bool, default=True, help='IJB-C Evaluation')
    parser.add_argument('--Verification_Test_Train', type=bool, default=True, help='MS1M/VGG2 Evaluation')
    # Analysis
    parser.add_argument('--TestCode', type=bool, default=False, help='test code or not')
    args = parser.parse_args()


    if args.Verification_Test_Train==False and args.Triplet_Loss==True:
        print('Arguments of verification testing and verification loss is mismatch')
        exit()

    # ======================================================
    # Hyper-parameters & data loaders
    #   contains base options
    # ======================================================
    cfg = configurations[1]


    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)
    TRAIN_DATA_ROOT = args.TRAIN_DATA_ROOT

    # Save information
    if cfg['TestCode']==False:
        args.MODEL_ROOT = '{}/{}'.format(cfg['MODEL_ROOT'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # the root to buffer your checkpoints
        LOG_ROOT = '{}/{}'.format(cfg['LOG_ROOT'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # the root to log your train/val status
        BACKBONE_RESUME_ROOT = args.BACKBONE_RESUME_ROOT # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = args.HEAD_RESUME_ROOT  # the root to resume training from a saved checkpoint

        if not os.path.exists(args.MODEL_ROOT): os.makedirs(args.MODEL_ROOT)
        if not os.path.exists(LOG_ROOT): os.makedirs(LOG_ROOT)
        # Save the parameter and model
        for attr, value in sorted(args.__dict__.items()):
            text = "\t{}={}\n".format(attr.upper(), value)
            with open('{}/Parameters.txt'.format(args.MODEL_ROOT), 'a') as f:
                f.write(text)
        shutil.copy('./head/metrics.py', '{}/metrics.py'.format(args.MODEL_ROOT))
        shutil.copy('./train.py', '{}/train.py'.format(args.MODEL_ROOT))
        shutil.copy('./config.py', '{}/config.py'.format(args.MODEL_ROOT))

        writer = SummaryWriter(LOG_ROOT)  # writer for buffering intermedium results



    # fool-proof the dataset
    if args.Verification_Test_IJBC:  foolproof_dataset('{}/{}.csv'.format(args.BASE_FILE_PATH, args.CSV_IJBC_FILE), args.TEST_IJBC_DATA_ROOT, file_name='IJBC')
    if args.Verification_Test_Train:
        args.verif_List = verification_list(args.BASE_FILE_PATH, args.CSV_VGGFace2_FILE)
        foolproof_dataset(args.verif_List['Test_List'], args.TRAIN_DATA_ROOT, file_name='MS1M')


    BACKBONE_RESUME_ROOT = args.BACKBONE_RESUME_ROOT # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = args.HEAD_RESUME_ROOT  # the root to resume training from a saved checkpoint
    BATCH_SIZE = args.BATCH_SIZE
    LR = args.LR  # initial LR
    NUM_EPOCH = args.NUM_EPOCH
    WEIGHT_DECAY = args.WEIGHT_DECAY
    MOMENTUM = args.MOMENTUM

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension

    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']



    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])


    dataset_train = datasets.ImageFolder(TRAIN_DATA_ROOT, train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )


    # Verification pairs
    if args.Verification_Test_Train:
        verification_dataset = FaceIdPoseDataset_Pairs(args.verif_List['Train_Pair'], TRAIN_DATA_ROOT,
                                          transform=transforms.Compose([
                                              transforms.Resize( [int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),
                                              transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
                                          ]))
        verification_dataloader = DataLoader(verification_dataset, batch_size=BATCH_SIZE , shuffle=True)  # , num_workers=6)
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    triplet_loss_batch_hard = TripletLoss_BatchHard(DEVICE, margin=args.lambda_triplet_margin)
    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))


    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                     'ResNet_101': ResNet_101(INPUT_SIZE), 
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE), 
                     'IR_101': IR_101(INPUT_SIZE), 
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                     'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]

    HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'MagFace': MagFace(feat_dim = EMBEDDING_SIZE, num_class = NUM_CLASS, device_id = GPU_ID),
                 }

    HEAD = HEAD_DICT[HEAD_NAME]

    LOSS_DICT = {'Focal': FocalLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]



    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)

    # optionally resume from a checkpoint
    if os.path.isfile(BACKBONE_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    if os.path.isfile(HEAD_RESUME_ROOT):
        print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
        HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
        OPTIMIZER = OPTIMIZER.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 100 # frequency to display training loss & acc
    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    

    # _ = Validation_VGGFace2(BACKBONE, cfg, args, DEVICE, 'ArcFace')
    # _ = Validation_IJBC_Covariate(BACKBONE, cfg, args, DEVICE, 'ArcFace')

    batch = 0  # batch index

    for epoch in range(NUM_EPOCH): # start training process
        
        if epoch == STAGES[0]: # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        BACKBONE.train()  # set to training mode
        HEAD.train()

        losses = AverageMeter()
        losses_arc = AverageMeter()
        losses_tri = AverageMeter()

        
        Save_dict = {'Pos_1':[], 'Pos_2':[], 'Pos_Score':[], 'Neg_1':[], 'Neg_2':[], 'Neg_Score':[], 'Final_Score':[]} 

        for inputs, labels in tqdm(iter(train_loader)):
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # verification loss
            if args.Triplet_Loss:
                anchor, positive, anchor_labels, positive_labels, anchor_names, positive_names = next(iter(verification_dataloader))
                anchor, positive, anchor_labels, positive_labels = anchor.to(DEVICE), positive.to(DEVICE) , anchor_labels.to(DEVICE).long(), positive_labels.to(DEVICE).long()
                anchor_feats, positive_feats = BACKBONE(anchor), BACKBONE(positive)
                # tri_loss, Save_dict = triplet_loss_batch_hard(anchor_feats, positive_feats, anchor_labels, positive_labels, anchor_names, positive_names, Save_dict)
                tri_loss = triplet_loss_batch_hard(anchor_feats, positive_feats, anchor_labels, positive_labels, anchor_names, positive_names)
                # tri_loss = triplet_loss(anchor_feats, positive_feats, negative_feats)

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            arc_loss = LOSS(outputs, labels)


            if args.Triplet_Loss:
                loss = arc_loss + args.lambda_triplet * tri_loss
                losses_tri.update(tri_loss.data.item(), inputs.size(0))
            else:
                loss = arc_loss

            losses.update(loss.data.item(), inputs.size(0))
            losses_arc.update(arc_loss.data.item(), inputs.size(0))
            if args.Triplet_Loss: losses_tri.update(tri_loss.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            batch += 1 # batch index

            writer.add_scalar('step/losses_all', loss, batch)
            writer.add_scalar('step/losses_arc', arc_loss, batch)
            if args.Triplet_Loss: writer.add_scalar('step/losses_tri', tri_loss, batch)
    

        # 
        for idx, value in enumerate(np.sort(list(Save_dict))):
            if idx == 0:
                save_info = np.array(Save_dict[value]).reshape(-1, 1)
            else:
                save_info = np.concatenate((save_info, np.array(Save_dict[value]).reshape(-1, 1)), axis=1)
        
        # 'Final_Score', 'Neg_1', 'Neg_2', 'Neg_Score', 'Pos_1', 'Pos_2', 'Pos_Score'
        if args.Triplet_Loss: np.save('{}/triplet_info_{}'.format(args.MODEL_ROOT, epoch + 1), save_info)


        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        losses_arc = losses_arc.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Arc_Loss", losses_arc, epoch + 1)
        if args.Triplet_Loss:
            losses_tri = losses_tri.avg
            writer.add_scalar("Tri_Loss", losses_tri, epoch + 1)


        # save checkpoints per epoch
        if (epoch + 1) % args.SAVE_FREQ == 0:

            if args.Verification_Test_Train: _ = Validation_VGGFace2(BACKBONE, cfg, args, DEVICE, (epoch + 1))
            if args.Verification_Test_IJBC: _ = Validation_IJBC_Covariate(BACKBONE, cfg, args, DEVICE, (epoch + 1))

            # save checkpoints per epoch
            if MULTI_GPU:
                torch.save(BACKBONE.module.state_dict(), os.path.join(args.MODEL_ROOT, "Backbone_{}_Epoch_{}.pth".format(BACKBONE_NAME, epoch + 1)), _use_new_zipfile_serialization=False)
                torch.save(HEAD.module.state_dict(), os.path.join(args.MODEL_ROOT, "Head_{}_Epoch_{}_.pth".format(HEAD_NAME, epoch + 1)), _use_new_zipfile_serialization=False)
            else:
                torch.save(BACKBONE.state_dict(), os.path.join(args.MODEL_ROOT, "Backbone_{}_Epoch_{}.pth".format(BACKBONE_NAME, epoch + 1)), _use_new_zipfile_serialization=False)
                torch.save(HEAD.state_dict(), os.path.join(args.MODEL_ROOT, "Head_{}_Epoch_{}.pth".format(BACKBONE_NAME, epoch + 1)), _use_new_zipfile_serialization=False)

