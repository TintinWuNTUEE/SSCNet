
import os
import argparse
import torch
import torch.nn as nn
import sys
import yaml
from losses.loss import panoptic_loss
from common.io_tools import dict_to
from common.instance_post_processing import get_panoptic_segmentation
from common.eval_pq import PanopticEval
import numpy as np
import common.checkpoint as checkpoint
from common.dataset import get_dataset
from common.config import CFG, merge_configs
from models.model import get_model
from common.logger import get_logger
import wandb
def get_mem_allocated(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
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

def train(model1, model2, optimizer, scheduler, dataset, _cfg, p_args, start_epoch, logger):
  device = torch.device('cuda')
  dtype = torch.float32

  model1 = model1.to(device)
  model2 = model2.to(device)
  
  
  best_loss = 99999999999
  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.to(device)

  dset = dataset['train']

  nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']
  loss_fn = panoptic_loss(center_loss_weight = p_args['model']['center_loss_weight'], offset_loss_weight = p_args['model']['offset_loss_weight'],\
                            center_loss = p_args['model']['center_loss'], offset_loss=p_args['model']['offset_loss'])
  for epoch in range(start_epoch, nbr_epochs+1):
    checkpoint_path = p_args['model']['model_save_path']
    checkpoint.save_last(checkpoint_path,model2,optimizer,scheduler,epoch)
    model1.train()
    model2.train()
    logger.info('=> =========== Epoch [{}/{}] ==========='.format(epoch, nbr_epochs))
    logger.info('=> Reminder - Output of routine on {}'.format(_cfg._dict['OUTPUT']['OUTPUT_PATH']))

    logger.info('=> Learning rate: {}'.format(scheduler.get_last_lr()[0]))
    for t, (data, _) in enumerate(dset):
      voxel_label = data['3D_LABEL'].type(torch.LongTensor).to(device).permute(0,1,3,2)
      data = dict_to(data, device, dtype)
      scores = model1(data)
      label,train_gt_center_tensor,train_gt_offset_tensor = data['PREPROCESS']
      train_gt_center_tensor,train_gt_offset_tensor =train_gt_center_tensor.to(device),train_gt_offset_tensor.to(device)
      # forward
      input_feature = scores['pred_semantic_1_1_feature'].view(-1,256,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
      sem_prediction,center,offset = model2(input_feature)
      # loss2
      loss = loss_fn(sem_prediction,center,offset,voxel_label,train_gt_center_tensor,train_gt_offset_tensor)
      # backward + optimize
      # gradient accumulator
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
      if t % 1000 == 0:
        logger.info ("LOSS:{}".format(loss.item()))
      wandb.log({"loss": loss})
      # Optional
      wandb.watch(model2)
    scheduler.step()
    best_loss = validation(model1, model2, optimizer,scheduler,loss_fn,dataset, _cfg,p_args,epoch, logger,best_loss)
    logger.info ("FINAL SUMMARY=>LOSS:{}".format(loss.item()))
    get_mem_allocated(device)

def validation(model1, model2, optimizer,scheduler, loss_fn,dataset, _cfg,p_args,epoch, logger, best_loss):
  device = torch.device('cuda')
  dtype = torch.float32  # Tensor type to be used
  nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']
  grid_size = p_args['dataset']['grid_size']  
  dset = dataset['val']
  logger.info('=> Passing the network on the validation set...')
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
      label,val_gt_center_tensor,val_gt_offset_tensor = data['PREPROCESS']
      
      val_gt_center_tensor,val_gt_offset_tensor =val_gt_center_tensor.to(device),val_gt_offset_tensor.to(device)
      loss1 = model1.compute_loss(scores, data)

      input_feature = scores['pred_semantic_1_1_feature'].view(-1,256,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
      sem_prediction,center,offset = model2(input_feature)

      # loss2
      loss2 = loss_fn(sem_prediction,center,offset,voxel_label,val_gt_center_tensor,val_gt_offset_tensor)
      panoptic_labels, _ = get_panoptic_segmentation(sem_prediction, center, offset, dset.dataset.thing_list,\
                                                                threshold=p_args['model']['post_proc']['threshold'], nms_kernel=p_args['model']['post_proc']['nms_kernel'],\
                                                                top_k=p_args['model']['post_proc']['top_k'], polar=p_args['model']['polar'])
      evaluator.addBatch(panoptic_labels & 0xFFFF, panoptic_labels, voxel_label)
      
      # backward + optimize
      loss = loss1['total']+loss2
      loss = loss.item()

      # for l_key in loss1:
      #   tbwriter.add_scalar('validation_loss_batch/{}'.format(l_key), loss1[l_key].item(), len(dset) * (epoch-1) + t)
      # Updating batch losses to then get mean for epoch loss

      if (t + 1) % _cfg._dict['VAL']['SUMMARY_PERIOD'] == 0:
        print('=> Epoch [{}/{}], Iteration [{}/{}], Val Losses:{} '.format(epoch, nbr_epochs, t+1, len(dset),loss))
    
    miou, ious = evaluator.getSemIoU()
    
    #logger validation score
      
    print('Validation per class IoU: ')
    logger.info('Validation per class IoU(Panoptic polarnet): ')
    for class_name, class_iou in zip(unique_label_str, ious[1:]):
      print('%15s : %6.2f%%'%(class_name, class_iou*100))
      logger.info('%15s : %6.2f%%'%(class_name, class_iou*100))
    print('Current val miou is %.3f'%(miou*100))
    logger.info(('Current val miou is %.3f'%(miou*100)))
    wandb.log({"miou": miou})
    checkpoint_path = None
    if loss<best_loss:
      best_loss=loss
      checkpoint_path=p_args['model']['best']
      checkpoint.save_panoptic(checkpoint_path,model2,optimizer,scheduler,epoch)
    checkpoint_path = p_args['model']['model_save_path']
    checkpoint.save_last(checkpoint_path,model2,optimizer,scheduler,epoch)
  return best_loss
def main():
  LMSC_args = parse_args('LMSCNet')
  p_args=parse_args('Panoptic Polarnet')
  train_f = LMSC_args.config_file
  dataset_f = LMSC_args.dataset_root
  
  wandb.init(project="SSCNET", entity="tintinwu")
  
  # Read train configuration file
  _cfg = CFG()
  _cfg.from_config_yaml(train_f)
  wandb.config = {
  "learning_rate": _cfg._dict['OPTIMIZER']['BASE_LR'],
  "epochs": 80,
  "batch_size": 4
}
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
  # lambda1 = lambda epoch: (0.98) ** (epoch)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
  
  model1= checkpoint.load_LMSC(model1,optimizer,scheduler,_cfg._dict['STATUS']['RESUME'],_cfg._dict['STATUS']['LAST'],logger)
  model2,optimizer,scheduler,epoch = checkpoint.load_panoptic(model2,scheduler,optimizer,p_args['model']['model_save_path'],logger)
  
  train(model1, model2, optimizer, scheduler, dataset, _cfg, p_args, epoch, logger)
  logger.info('=> ============ Network trained - all epochs passed... ============')
  exit()
if __name__ == '__main__':
  main()
