from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import os
from glob import glob



def load(model, optimizer, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if os.path.isfile(path):
    checkpoint = torch.load(path)
    epoch = checkpoint.pop('startEpoch')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
      model.module.load_state_dict(checkpoint.pop('model'))
    else:
      model.load_state_dict(checkpoint.pop('model'))
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model, optimizer,  epoch
  else:
    logger.info('=> No checkpoint. Initializing model from scratch')
    epoch = 1
    return model, optimizer,  epoch


def save(path, model, optimizer, epoch):
  '''
  Save checkpoint file
  '''

  torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
  }, path)
