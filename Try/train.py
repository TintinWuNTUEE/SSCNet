from dataset import get_dataset
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import yaml

from metrics import Metrics
from io_tools import dict_to


def parse_args():
  parser = argparse.ArgumentParser(description='LMSCNet + Panoptic_PolarNet training')

  parser.add_argument('-d', '--data_dir', help='path to dataset root folder', default='../semanticKITTI/dataset')
  parser.add_argument('-p', '--model_save_path', default='./weights/Panoptic_SemKITTI.pt')
  parser.add_argument('-c', '--configs', help='path to config file', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')
  parser.add_argument('--pretrained_model', default='empty')
  args = parser.parse_args()

  return args

def train(model1, model2, optimizer, scheduler, dataset, _cfg, start_epoch, logger, tbwriter):
  device = torch.device('cuda')
  dtype = torch.float32

  model1 = model1.to(device)
  model2 = model2.to(device)

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
      scores = model1(data)
      loss1 = model1.compute_loss(scores, data)
      
      optimizer.zero_grad()

      loss1['total'].backward()

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

      



def main():
  args = parse_args()

if __name__ == '__main__':
  main()
