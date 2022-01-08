import argparse
import torch
import yaml
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from common.io_tools import dict_to
from dataloader.dataset import get_dataset
from common.configs import merge_configs
from common.utils import get_instance, get_unique_label, get_lr
from common.logger import get_logger
from common.iou import iou_pytorch, iou_numpy, iou
from models.Unet import Unet, SegmentationHead
from common.checkpoint import save, load
from loss import BinaryFocalLossWithLogits, FocalLoss
############################## grid size setting ##############################
max_bound = np.asarray([51.2,25.6,4.4])
min_bound = np.asarray([0,-25.6,-2])
crop_range = max_bound-min_bound
cur_grid_size = np.asarray([256,256,32])
intervals = crop_range/(cur_grid_size-1)
############################## grid size setting ##############################
# Seed
seed = 321
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def get_mem_allocated(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--data_dir', help='path to dataset root folder', default='../semanticKITTI/dataset')
    parser.add_argument('-p', '--model_save_path', default='./weights')
    parser.add_argument('-c', '--configs', help='path to config file', default='configs/Panoptic-PolarNet.yaml')
    parser.add_argument('--pretrained_model', default='empty')
    args = parser.parse_args()
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args,new_args)
  
    return args
def train(model,loss_fn,scheduler,optimizer,dataset,args,logger,start_epoch=0):
    device=torch.device('cuda')
    dset = dataset['train']
    nbr_epochs = args['model']['max_epoch']
    sample_type = args['dataset']['type']
    dtype = torch.float32
    model.to(device)
    model.train()
    for epoch in range(start_epoch,nbr_epochs+1):
        epoch_loss = []
        epoch_iou = []
        x = 0
        for i,(input_pos,input_class,label_pos,label_class)  in enumerate(dset):
            
            # for j in range(input_class.shape[0]):
                # print(x)
                # fig = plt.figure(x)
                # partial_sem = input_class[j]&0xffff
                # partial_inst = (input_class[j]&0xffff0000)>>16
                # complete_sem = label_class[j]&0xffff
                # complete_inst = (label_class[j]&0xffff0000)>>16

                # voxel1 = fig.add_subplot(121,projection='3d')
                # voxel1.voxels(input_pos[j].squeeze())
                # voxel1.title.set_text(" partial_sem : "+str(partial_sem)+" inst : "+str(partial_inst))
                # voxel2 = fig.add_subplot(122,projection='3d')
                # voxel2.voxels(label_pos[j].squeeze())
                # voxel2.title.set_text(" complete_sem : "+str(complete_sem)+" inst : "+str(complete_inst))
                # x += 1
            # print(input_pos.sum((2,3,4)))
            # print([i & 0xffff for i in input_class])
            # print(label_pos.sum((2,3,4)))
            # print([i & 0xffff for i in label_class])
            # print(input_class.shape)
            # print(iou(label_pos,label_pos, n_classes=1))
            # return
            # print(i)
            input_pos,input_class,label_pos,label_class= input_pos.to(device),input_class.to(device),label_pos.to(device),label_class.to(device)
            # print(input_pos.shape)
            
            pred = model(input_pos)
            # print(pred.shape)
            # print(label_pos.shape)
            # print(pred.type())
            # print(label_pos.type())
            loss = loss_fn(pred,label_pos.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                epoch_loss += [loss.item()]
                epoch_iou += iou(pred, label_pos, n_classes=1)
                # print(loss.item())
        # print(epoch_iou)
        # print('done')
        # plt.show()
        # return
        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        epoch_iou = sum(epoch_iou)/len(epoch_iou)
        logger.info('=> [Epoch {} - Total Train Loss = {}, Total Train IOU = {}]'.format(epoch, epoch_loss, epoch_iou))
        logger.info('lr : {}'.format(get_lr(optimizer)))
        # scheduler.step()
    save(args['model']['voxel_instance_model_save_path'],'voxel_instance.pt',model,optimizer,epoch,args)

def validation(model,loss_fn,dataset,args,logger,start_epoch=0):
    device=torch.device('cuda')
    dset = dataset['val']
    model.to(device)
    epoch_loss = []
    epoch_iou = []
    with torch.no_grad():
        for i,(input_pos,input_class,label_pos,label_class)  in enumerate(dset):
            # print(i)
            input_pos,input_class,label_pos,label_class= input_pos.to(device),input_class.to(device),label_pos.to(device),label_class.to(device)
            # print(input_pos.shape)
            
            pred = model(input_pos)
            # print(pred.shape)
            # print(label_pos.shape)
            # print(pred.type())
            # print(label_pos.type())
            tosave = (pred.argmax(dim=1).squeeze(),input_class, label_pos.squeeze(),label_class)
            torch.save(tosave,'object1_pred')
            break
        #     loss = loss_fn(pred,label_pos.squeeze(axis=1).long())
        #     epoch_loss += [loss.item()]
        #     epoch_iou += [iou_pytorch(pred, label_pos).mean()]
        #     # print(loss.item())
        # epoch_loss = sum(epoch_loss)/len(epoch_loss)
        # epoch_iou = sum(epoch_iou)/len(epoch_iou)
        # logger.info('=> [ Total Validation Loss = {}, Total Validation IOU = {} ]'.format(epoch_loss, epoch_iou))
        # logger.info('lr : {}'.format(get_lr(optimizer)))
        return
        

if __name__ == '__main__':
    args = parse_args()
    dataset=get_dataset(args)
    # model = Unet(out_channel=1)
    model = SegmentationHead(1,2,1,[1,2,3])
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = FocalLoss(2)
    loss_fn = BinaryFocalLossWithLogits(0.25,2.,'mean')
    # optimizer = optim.SGD(model.parameters(),lr=args['TRAIN']['learning_rate'])
    optimizer = optim.Adam(model.parameters(),lr=args['TRAIN']['learning_rate'])
    lambda1 = lambda epoch: (0.98) ** (epoch)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler = None
    logger = get_logger(args['model']['train_log'],'voxel_train.log')
    train(model,loss_fn,scheduler,optimizer,dataset,args,logger)
    # validation(model,loss_fn,dataset,args,logger)