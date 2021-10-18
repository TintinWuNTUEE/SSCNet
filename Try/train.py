
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import yaml
from losses.loss import panoptic_loss
from common.metrics import Metrics
from common.io_tools import dict_to
from common.instance_post_processing import get_panoptic_segmentation
from common.eval_pq import PanopticEval
import numpy as np
import common.checkpoint as checkpoint
from common.dataset import SemanticKITTI,get_dataset
from common.config import CFG, merge_configs
from models.model import get_model
from common.logger import get_logger
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
    parser.add_argument('-p', '--model_save_path', default='./weights/Panoptic_SemKITTI.pt')
    parser.add_argument('-c', '--configs', help='path to config file', default='configs/Panoptic-PolarNet.yaml')
    parser.add_argument('--pretrained_model', default='empty')
    args = parser.parse_args()
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args,new_args)
  
  return args

def train(model1, model2, optimizer, scheduler, dataset, _cfg, p_args, start_epoch, logger, tbwriter):
  device = torch.device('cuda')
  dtype = torch.float32

  model1 = model1.to(device)
  model2 = model2.to(device)
  best_loss = 999999999999
  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.to(device)

  dset = dataset['train']

  nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']
  nbr_iterations = len(dset)
  loss_fn = panoptic_loss(center_loss_weight = p_args['model']['center_loss_weight'], offset_loss_weight = p_args['model']['offset_loss_weight'],\
                            center_loss = p_args['model']['center_loss'], offset_loss=p_args['model']['offset_loss'])
  # Defining metrics class and initializing them..
  metrics = Metrics(dset.dataset.nbr_classes, nbr_iterations, model1.get_scales())
  metrics.reset_evaluator()
  metrics.losses_track.set_validation_losses(model1.get_validation_loss_keys())
  metrics.losses_track.set_train_losses(model1.get_train_loss_keys())
  for epoch in range(start_epoch, nbr_epochs+1):
    
    model1.train()
    model2.train()
    logger.info('=> =========== Epoch [{}/{}] ==========='.format(epoch, nbr_epochs))
    logger.info('=> Reminder - Output of routine on {}'.format(_cfg._dict['OUTPUT']['OUTPUT_PATH']))

    logger.info('=> Learning rate: {}'.format(scheduler.get_lr()[0]))
    for t, (data, indices) in enumerate(dset):
      logger.info(t)
      data = dict_to(data, device, dtype)
      train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor = data['PREPROCESS']
      train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor = train_label_tensor.type(torch.LongTensor).to(device),train_gt_center_tensor.to(device),train_gt_offset_tensor.to(device)
      scores = model1(data)
      loss1 = model1.compute_loss(scores, data)
      # forward
      input = scores['pred_semantic_1_1'].view(-1,640,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
      input_feature = scores['pred_semantic_1_1_feature'].view(-1,256,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
      print('fitting into model2')
      sem_prediction,center,offset = model2(input_feature)
      # loss2
      loss2 = loss_fn(sem_prediction,center,offset,train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor)
        # backward + optimize
      loss = loss1['total']+loss2
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if _cfg._dict['SCHEDULER']['FREQUENCY'] == 'iteration':
        scheduler.step()

      for loss_key in loss1:
        tbwriter.add_scalar('train_loss_batch/{}'.format(loss_key), loss1[loss_key].item(), len(dset)*(epoch-1)+t)
      # Updating batch losses to metric then get mean of epoch loss
      metrics.losses_track.update_train_losses(loss1)

      if (t+1) % _cfg._dict['TRAIN']['SUMMARY_PERIOD'] == 0:
        print('=> Epoch [{}/{}], Iteration [{}/{}], Learn Rate: {}, Train Losses:{} '.format(epoch, nbr_epochs, t+1, len(dset), scheduler.get_lr()[0],loss))

      metrics.add_batch(prediction=scores,target=model1.get_target(data))
      
    best_loss = validation(model1, model2, optimizer,scheduler,loss_fn,dataset, _cfg,p_args,epoch, logger,tbwriter,metrics,best_loss)
      
      
    for l_key in metrics.losses_track.train_losses:
          tbwriter.add_scalar('train_loss_epoch/{}'.format(l_key),
                          metrics.losses_track.train_losses[l_key].item()/metrics.losses_track.train_iteration_counts,
                          epoch - 1)
    tbwriter.add_scalar('lr/lr', scheduler.get_lr()[0], epoch - 1)

    epoch_loss = metrics.losses_track.train_losses['total']/metrics.losses_track.train_iteration_counts

    for scale in metrics.evaluator.keys():
      tbwriter.add_scalar('train_performance/{}/mIoU'.format(scale), metrics.get_semantics_mIoU(scale).item(), epoch-1)
      tbwriter.add_scalar('train_performance/{}/IoU'.format(scale), metrics.get_occupancy_IoU(scale).item(), epoch-1)
    logger.info('=> [Epoch {} - Total Train Loss = {}]'.format(epoch, epoch_loss))
    for scale in metrics.evaluator.keys():
      loss_scale = metrics.losses_track.train_losses['semantic_{}'.format(scale)].item()/metrics.losses_track.train_iteration_counts
      logger.info('=> [Epoch {} - Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                  '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'
                  .format(epoch, scale, loss_scale,
                          metrics.get_semantics_mIoU(scale).item(),
                          metrics.get_occupancy_IoU(scale).item(),
                          metrics.get_occupancy_Precision(scale).item(),
                          metrics.get_occupancy_Recall(scale).item(),
                          metrics.get_occupancy_F1(scale).item()))

    logger.info('=> Epoch {} - Training set class-wise IoU:'.format(epoch))
    for i in range(1, metrics.nbr_classes):
      class_name  = dset.dataset.dataset_config['labels'][dset.dataset.dataset_config['learning_map_inv'][i]]
      class_score = metrics.evaluator['1_1'].getIoU()[1][i]
      logger.info('=> IoU {}: {:.6f}'.format(class_name, class_score))

    # Reset evaluator and losses for next epoch...
    metrics.reset_evaluator()
    metrics.losses_track.restart_train_losses()
    metrics.losses_track.restart_validation_losses()

    if _cfg._dict['SCHEDULER']['FREQUENCY'] == 'epoch':
      scheduler.step()
    
    _cfg.update_config(resume=True)
    logger.info ("FINAL SUMMARY=>LOSS:{}".format(loss.item()))
    
def validation(model1, model2, optimizer,scheduler, loss_fn,dataset, _cfg,p_args,epoch, logger, tbwriter,metrics,best_loss):
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

    for t, (data, indices) in enumerate(dset):
      val_label_tensor,val_gt_center_tensor,val_gt_offset_tensor = data['PREPROCESS']
      val_label_tensor,val_gt_center_tensor,val_gt_offset_tensor = val_label_tensor.type(torch.LongTensor).to(device),val_gt_center_tensor.to(device),val_gt_offset_tensor.to(device)
      # mask will be done in eval.py when foreground is none
      # for_mask = torch.zeros(1,grid_size[0],grid_size[1],grid_size[2],dtype=torch.bool).to(device)
      # for_mask[(val_label_tensor>=0 )& (val_label_tensor<8)] = True 
      
      data= dict_to(data, device, dtype)

      scores = model1(data)
      loss1 = model1.compute_loss(scores, data)
      
      input = scores['pred_semantic_1_1'].view(-1,640,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
      input_feature = scores['pred_semantic_1_1_feature'].view(-1,256,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
      
      sem_prediction,center,offset = model2(input_feature)
      # loss2
      loss2 = loss_fn(sem_prediction,center,offset,val_label_tensor,val_gt_center_tensor,val_gt_offset_tensor)
      panoptic_labels, center_points = get_panoptic_segmentation(sem_prediction, center, offset, dset.dataset.thing_list,\
                                                                threshold=p_args['model']['post_proc']['threshold'], nms_kernel=p_args['model']['post_proc']['nms_kernel'],\
                                                                top_k=p_args['model']['post_proc']['top_k'], polar=p_args['model']['polar'])
      evaluator.addBatch(panoptic_labels & 0xFFFF, panoptic_labels, val_label_tensor)
      
      # backward + optimize
      loss = loss1['total']+loss2


      for l_key in loss1:
        tbwriter.add_scalar('validation_loss_batch/{}'.format(l_key), loss1[l_key].item(), len(dset) * (epoch-1) + t)
      # Updating batch losses to then get mean for epoch loss
      metrics.losses_track.update_validaiton_losses(loss1)

      if (t + 1) % _cfg._dict['VAL']['SUMMARY_PERIOD'] == 0:
        print('=> Epoch [{}/{}], Iteration [{}/{}], Val Losses:{} '.format(epoch, nbr_epochs, t+1, len(dset),loss))

      metrics.add_batch(prediction=scores, target=model1.get_target(data))

    for l_key in metrics.losses_track.validation_losses:
      tbwriter.add_scalar('validation_loss_epoch/{}'.format(l_key),
                          metrics.losses_track.validation_losses[l_key].item()/metrics.losses_track.validation_iteration_counts,
                          epoch - 1)

    epoch_loss = metrics.losses_track.validation_losses['total']/metrics.losses_track.validation_iteration_counts

    for scale in metrics.evaluator.keys():
      tbwriter.add_scalar('validation_performance/{}/mIoU'.format(scale), metrics.get_semantics_mIoU(scale).item(), epoch-1)
      tbwriter.add_scalar('validation_performance/{}/IoU'.format(scale), metrics.get_occupancy_IoU(scale).item(), epoch-1)

    logger.info('=> [Epoch {} - Total Validation Loss = {}]'.format(epoch, epoch_loss))
    for scale in metrics.evaluator.keys():
      loss_scale = metrics.losses_track.validation_losses['semantic_{}'.format(scale)].item()/metrics.losses_track.train_iteration_counts
      logger.info('=> [Epoch {} - Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                  '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'
                  .format(epoch, scale, loss_scale,
                          metrics.get_semantics_mIoU(scale).item(),
                          metrics.get_occupancy_IoU(scale).item(),
                          metrics.get_occupancy_Precision(scale).item(),
                          metrics.get_occupancy_Recall(scale).item(),
                          metrics.get_occupancy_F1(scale).item()))
    
    miou, ious = evaluator.getSemIoU()
      
    print('Validation per class IoU: ')
    logger.info('Validation per class IoU: ')
    for class_name, class_iou in zip(unique_label_str, ious[1:]):
      print('%15s : %6.2f%%'%(class_name, class_iou*100))
      logger.info('%15s : %6.2f%%'%(class_name, class_iou*100))
    print('Current val miou is %.3f'%(miou*100))
    logger.info(('Current val miou is %.3f'%(miou*100)))


    logger.info('=> Epoch {} - Validation set class-wise IoU:'.format(epoch))
    for i in range(1, metrics.nbr_classes):
      class_name  = dset.dataset.dataset_config['labels'][dset.dataset.dataset_config['learning_map_inv'][i]]
      class_score = metrics.evaluator['1_1'].getIoU()[1][i]
      logger.info('    => {}: {:.6f}'.format(class_name, class_score))

    checkpoint_info = {}

    if epoch_loss < _cfg._dict['OUTPUT']['BEST_LOSS']:
      logger.info('=> Best loss on validation set encountered: ({} < {})'.
                  format(epoch_loss, _cfg._dict['OUTPUT']['BEST_LOSS']))
      _cfg._dict['OUTPUT']['BEST_LOSS'] = epoch_loss.item()
      checkpoint_info['best-loss'] = 'BEST_LOSS'

    mIoU_1_1 = metrics.get_semantics_mIoU('1_1')
    IoU_1_1  = metrics.get_occupancy_IoU('1_1')
    if mIoU_1_1 > _cfg._dict['OUTPUT']['BEST_METRIC']:
      logger.info('=> Best metric on validation set encountered: ({} > {})'.
                  format(mIoU_1_1, _cfg._dict['OUTPUT']['BEST_METRIC']))
      _cfg._dict['OUTPUT']['BEST_METRIC'] = mIoU_1_1.item()
      checkpoint_info['best-metric'] = 'BEST_METRIC'
      metrics.update_best_metric_record(mIoU_1_1, IoU_1_1, epoch_loss.item(), epoch)
    if loss.item()<best_loss:
      best_loss=loss.item()
      checkpoint_path = os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'chkpt', str(epoch).zfill(2))
      checkpoint.save_LMSC(checkpoint_path, model1, optimizer, scheduler, epoch, _cfg._dict)
      checkpoint.save_panoptic(p_args['model']['model_save_path'],model2,optimizer,epoch)
  return best_loss
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

  tbwriter = SummaryWriter(log_dir=os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'metrics'))
  logger = get_logger(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'logs_train.log')
  logger.info('============ Training routine: "%s" ============\n' % train_f)
  #get dataset(dataset.py)
  dataset = get_dataset(_cfg)
  
  #get model(model.py)
  logger.info('=> Loading network architecture...')
  model1,model2 = get_model(_cfg, dataset['train'].dataset)
  
  #build optimizer
  logger.info('=> Loading optimizer...')
  params = list(model1.get_parameters())+list(model2.parameters())
  optimizer = torch.optim.Adam(params,lr=_cfg._dict['OPTIMIZER']['BASE_LR'],betas=(0.9, 0.999))
  
  #build scheduler
  logger.info('=> Loading scheduler...')
  lambda1 = lambda epoch: (0.98) ** (epoch)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
  
  model1,optimizer,scheduler,epoch = checkpoint.load_LMSC(model1,optimizer,scheduler,_cfg._dict['STATUS']['RESUME'],_cfg._dict['STATUS']['LAST'],logger)
  model2,optimizer,epoch = checkpoint.load_panoptic(model2,optimizer,p_args['model']['model_save_path'],logger)
  
  train(model1, model2, optimizer, scheduler, dataset, _cfg, p_args, epoch, logger, tbwriter)
  logger.info('=> ============ Network trained - all epochs passed... ============')
  exit()
if __name__ == '__main__':
  main()
