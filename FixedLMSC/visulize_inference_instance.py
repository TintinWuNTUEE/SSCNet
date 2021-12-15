import os
import argparse
import torch
import torch.nn as nn
import sys
import yaml
from losses.loss import panoptic_loss
from common.io_tools import dict_to
from common.eval_pq import PanopticEval
import numpy as np
import common.checkpoint as checkpoint
from common.dataset import get_dataset
from common.config import CFG, merge_configs
from models.model import get_model
from common.logger import get_logger
import matplotlib.pyplot as plt

def parse_args(modelname):
    parser = argparse.ArgumentParser(description='Training')
    if modelname == 'LMSCNet':
        parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/LMSCNet_SS.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
        )
        parser.add_argument(
        '--dset_root',
        dest='dataset_root',
        default=None,
        metavar='DATASET',
        help='path to dataset root folder',
        type=str,
        )
        args = parser.parse_args()
    elif modelname == 'Panoptic Polarnet':
        parser.add_argument('-d', '--data_dir', help='path to dataset root folder', default='../semanticKITTI/dataset')
        parser.add_argument('-p', '--model_save_path', default='./weights')
        parser.add_argument('-c', '--configs', help='path to config file', default='configs/Panoptic-PolarNet.yaml')
        parser.add_argument('--pretrained_model', default='empty')
        args = parser.parse_args()
        with open(args.configs, 'r') as s:
            new_args = yaml.safe_load(s)
        args = merge_configs(args,new_args)
    
    return args

def validation(model1, model2,dataset,_cfg,p_args):
    device = torch.device('cuda')
    dtype = torch.float32  # Tensor type to be used
    model1.to(device)
    model2.to(device)
    nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']
    grid_size = p_args['dataset']['grid_size']  
    dset = dataset['val']
    #evaluator = PanopticEval(len(unique_label)+1, None, [0], min_points=50)
    evaluator = PanopticEval(20+1, None, [0], min_points=50)
    
    #prepare miou fun  
    SemKITTI_label_name = dict()
    for i in sorted(list(dset.dataset.dataset_config['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[dset.dataset.dataset_config['learning_map'][i]] = dset.dataset.dataset_config['labels'][i]
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():

        for t, (data,_) in enumerate(dset):
        
            # mask will be done in eval.py when foreground is none
            # for_mask = torch.zeros(1,grid_size[0],grid_size[1],grid_size[2],dtype=torch.bool).to(device)
            # for_mask[(val_label_tensor>=0 )& (val_label_tensor<8)] = True 
            voxel_label = data['3D_LABEL'].type(torch.LongTensor).to(device).permute(0,1,3,2)
            data= dict_to(data, device, dtype)
            scores = model1(data)
            _,val_gt_center_tensor,val_gt_offset_tensor = data['PREPROCESS']
            val_gt_center_tensor,val_gt_offset_tensor =val_gt_center_tensor.to(device),val_gt_offset_tensor.to(device)
            loss1 = model1.compute_loss(scores, data)

            input_feature = scores['pred_semantic_1_1_feature'].view(-1,256,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
            first_stage_sem_prediction = scores['pred_semantic_1_1'].view(-1,32,256,256)
            sem_prediction,center,offset = model2(input_feature)
            
            first_stage_sem_prediction = first_stage_sem_prediction.cpu().numpy()
            sem_prediction,center,offset = sem_prediction.cpu().numpy(),center.cpu().numpy(), offset.cpu().numpy()
            
            first_stage_sem_prediction = first_stage_sem_prediction.argmax(axis=0).squeeze()
            sem_prediction = sem_prediction.argmax(axis=1).squeeze()
            center = center.squeeze()
            offset = offset.squeeze()
            
            print(first_stage_sem_prediction.shape)
            print(sem_prediction.shape)
            print(center.shape)
            print(offset.shape)
            
            
            thing_list = [i for i in dset.dataset.dataset_config['thing_class'].keys() if dset.dataset.dataset_config['thing_class'][i]==True]
            mask2 = np.zeros_like(first_stage_sem_prediction,dtype=bool)
            for label in thing_list:
                mask2[first_stage_sem_prediction == label] = True
                
            print(mask2.sum())
            first_stage_sem_prediction[~mask2] = 0
            first_stage_sem_prediction_bev = ((first_stage_sem_prediction>0).sum(axis=0))>0
            plot0 = plt.figure('first_stage_sem_prediction')
            plt.imshow(first_stage_sem_prediction_bev,cmap=plt.cm.gray,origin='lower')
            first_stage_sem_prediction_bev_nonzero = np.nonzero(first_stage_sem_prediction_bev)
            # print(partial_voxel_bev_nonzero)
            for row, col in zip(first_stage_sem_prediction_bev_nonzero[0],first_stage_sem_prediction_bev_nonzero[1]):
                for i in range(32):
                    if first_stage_sem_prediction[i,row, col] != 0:
                        # plt.text(col, row, str(data[row, col, i]), color='green',fontsize=12)
                        plt.text(col, row, str(first_stage_sem_prediction[i,row, col]), color='red',fontsize=12)
                        break
            
            
            mask1 = np.zeros_like(sem_prediction,dtype=bool)
            for label in thing_list:
                mask1[sem_prediction == label] = True
            print(mask1.sum())
            sem_prediction[~mask1] = 0
            sem_prediction_bev = ((sem_prediction>0).sum(axis=2))>0
            plot0 = plt.figure('sem_prediction_bev')
            plt.imshow(sem_prediction_bev,cmap=plt.cm.gray,origin='lower')
            sem_prediction_bev_nonzero = np.nonzero(sem_prediction_bev)
            # print(partial_voxel_bev_nonzero)
            for row, col in zip(sem_prediction_bev_nonzero[0],sem_prediction_bev_nonzero[1]):
                for i in range(32):
                    if sem_prediction[row, col, i] != 0:
                        # plt.text(col, row, str(data[row, col, i]), color='green',fontsize=12)
                        plt.text(col, row, str(sem_prediction[row, col, i]), color='red',fontsize=12)
                        break

            plot1 = plt.figure('center heat map')
            plt.imshow(center.squeeze(), cmap='hot', origin='lower')
            
            plot2, ax = plt.subplots()
            a = ax.quiver(offset[1,:,:],offset[0,:,:],angles='xy',scale_units='xy',scale=1)

            # label_to_be_save= (sem_prediction.cpu().numpy(),center.cpu().numpy(), offset.cpu().numpy())
            # folder_path = "../semanticKITTI/dataset/sequences/08/preprocess_"
            # filename = "000000.pt"
            # if not os.path.exists(folder_path):
            #     print(folder_path)
            #     os.makedirs(folder_path)
            # save_path = os.path.join(folder_path,filename)
            # torch.save(label_to_be_save,save_path)
            # print(t)
            plt.show()

            return

    
    return 

def main():
  LMSC_args = parse_args('LMSCNet')
  p_args=parse_args('Panoptic Polarnet')
  train_f = LMSC_args.config_file
  dataset_f = LMSC_args.dataset_root
  
  
  # Read train configuration file
  _cfg = CFG()
  _cfg.from_config_yaml(train_f)
  if dataset_f is not None:
    _cfg._dict['DATASET']['ROOT_DIR'] = dataset_f
  logger = get_logger(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'logs_train.log')
  logger.info('============ Training routine: "%s" ============\n' % train_f)
  #get dataset(dataset.py)
  dataset = get_dataset(_cfg)
  
  #get model(model.py)
  logger.info('=> Loading network architecture...')
  model1,model2 = get_model(_cfg, dataset['train'].dataset)
  
  #build optimizer
  logger.info('=> Loading optimizer...')
  params = model2.parameters()
  optimizer = torch.optim.Adam(params,lr=_cfg._dict['OPTIMIZER']['BASE_LR'],betas=(0.9, 0.999))
  
  #build scheduler
  logger.info('=> Loading scheduler...')
  lambda1 = lambda epoch: (0.98) ** (epoch)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
  
  model1= checkpoint.load_LMSC(model1,optimizer,scheduler,_cfg._dict['STATUS']['RESUME'],_cfg._dict['STATUS']['LAST'],logger)
  model2,optimizer,scheduler,epoch = checkpoint.load_panoptic(model2,scheduler,optimizer,p_args['model']['model_save_path'],logger)
  validation(model1, model2,dataset,_cfg,p_args)
  logger.info('=> ============ Network trained - all epochs passed... ============')
  exit()
if __name__ == '__main__':
  main()
