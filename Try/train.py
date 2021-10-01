from dataset import get_dataset
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import yaml
from loss import panoptic_loss
from metrics import Metrics
from io_tools import dict_to


def parse_args():
  parser = argparse.ArgumentParser(description='LMSCNet + Panoptic_PolarNet training')

  parser.add_argument('-d', '--data_dir', help='path to dataset root folder', default='../semanticKITTI/dataset')
  parser.add_argument('-p', '--model_save_path', default='./weights/Panoptic_SemKITTI.pt')
  parser.add_argument('-pc', '--panoptic_configs', help='path to config file', default='Panoptic-PolarNet.yaml')
  parser.add_argument('-lc', '--lmsc_configs', help='path to config file', default='LMSCNet_SS.yaml')
  parser.add_argument('--pretrained_model', default='empty')
  args = parser.parse_args()

  return args

def train(model1, model2, optimizer, scheduler, dataset, _cfg, p_args, start_epoch, logger, tbwriter):
  device = torch.device('cuda')
  dtype = torch.float32

  model1 = model1.to(device)
  model2 = model2.to(device)
  model1.train()
  model2.train()
  loss_fn = panoptic_loss(center_loss_weight = p_args['model']['center_loss_weight'], offset_loss_weight = p_args['model']['offset_loss_weight'],\
                            center_loss = p_args['model']['center_loss'], offset_loss=p_args['model']['offset_loss'])
  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.to(device)

  dataset = dataset['train']

  nbr_epochs = _cfg['TRAIN']['EPOCHS']
  nbr_iterations = len(dataset)

  # Defining metrics class and initializing them..
  metrics = Metrics(_cfg['nbr_classes'], nbr_iterations, model1.get_scales())
  metrics.reset_evaluator()
  metrics.losses_track.set_validation_losses(model1.get_validation_loss_keys())
  metrics.losses_track.set_train_losses(model1.get_train_loss_keys())

  for epoch in range(start_epoch, nbr_epochs+1):
    logger.info('=> =========== Epoch [{}/{}] ==========='.format(epoch, nbr_epochs))
    logger.info('=> Reminder - Output of routine on {}'.format(_cfg['OUTPUT']['OUTPUT_PATH']))

    logger.info('=> Learning rate: {}'.format(scheduler.get_lr()[0]))

    for t, (data, indices) in enumerate(dataset):
      data = dict_to(data, device, dtype)
      train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor = data['PREPROCESS']
      scores = model1(data)
      loss1 = model1.compute_loss(scores, data)
      
      sem_prediction,center,offset = model2(scores['pred_semantic_1_1_feature'])
      loss2 = loss_fn(sem_prediction,center,offset,train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor)
      #i think it wont work since here we got voxel but originally they prun points, they use pointnet so doesnt matter but we matter
      if p_args['model']['enable_SAP'] and epoch>=p_args['model']['SAP']['start_epoch']:
        loss.backward()
        for i,fea in enumerate(scores['pred_semantic_1_1_feature']):
          fea_grad = torch.norm(fea.grad,dim=1)
          top_k_grad, _ = torch.topk(fea_grad, int(p_args['model']['SAP']['rate']*fea_grad.shape[0]))
          # delete high influential points
          scores['pred_semantic_1_1_feature'][i] = scores['pred_semantic_1_1_feature'][i][fea_grad < top_k_grad[-1]]
        optimizer.zero_grad()
        # second pass
        # forward
        sem_prediction,center,offset = model2(scores)
                    # loss
        loss2 = loss_fn(sem_prediction,center,offset,train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor)
                    
        # backward + optimize
      loss = loss1['total']+loss2
      
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if _cfg['SCHEDULER']['FREQUENCY'] == 'iteration':
        scheduler.step()

      for loss_key in loss1:
        tbwriter.add_scalar('train)_loss_batch/{}'.format(loss_key), loss1[loss_key].item(), len(dataset)*(epoch-1)+t)
      # Updating batch losses to metric then get mean of epoch loss
      metrics.losses_track.update_train_losses(loss1)

      if (t+1) % _cfg['TRAIN']['SUMMARY_PERIOD'] == 0:
        print_loss = '=> Epoch [{}/{}], Iteration [{}/{}], Learn Rate: {}, Train Losses: '\
          .format(epoch, nbr_epochs, t+1, len(dataset), scheduler.get_lr()[0])
        for key in loss1.keys():
          print_loss += '{} = {:.6f}, '.format(key, loss1[key])
          logger.info(print_loss[:-3])

      metrics.add_batch(prediction=scores,target=model1.get_target(data))
      
      
      
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
      class_name  = dataset.dataset_config['labels'][dataset.dataset_config['learning_map_inv'][i]]
      class_score = metrics.evaluator['1_1'].getIoU()[1][i]
      logger.info('=> IoU {}: {:.6f}'.format(class_name, class_score))

    # Reset evaluator and losses for next epoch...
    metrics.reset_evaluator()
    metrics.losses_track.restart_train_losses()
    metrics.losses_track.restart_validation_losses()

    if _cfg._dict['SCHEDULER']['FREQUENCY'] == 'epoch':
      scheduler.step()

    _cfg.update_config(resume=True)
    
    
def validation(model1, model2, dataset, _cfg, p_args, epoch, logger, tbwriter, metrics):
  pass
  
    

def main():
  args = parse_args()

if __name__ == '__main__':
  main()
