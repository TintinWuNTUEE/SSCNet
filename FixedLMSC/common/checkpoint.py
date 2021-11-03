
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import os
from glob import glob
from common.io_tools import _remove_recursively, _create_directory
def load_LMSC(model, optimizer, scheduler, resume, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if not resume:
    logger.info('=> No checkpoint. Initializing model from scratch')
    model.weights_init()
    epoch = 1
    return model, optimizer, scheduler, epoch

  # If resume, check that path exists and load everything to return
  else:
    file_path = sorted(glob(os.path.join(path, '*.pth')))[0]
    assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
    checkpoint = torch.load(file_path, map_location='cpu')
    # epoch = checkpoint.pop('startEpoch')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
          model.module.load_state_dict(checkpoint.pop('model'))
    else:
      model.load_state_dict(checkpoint.pop('model'))
    # optimizer.load_state_dict(checkpoint.pop('optimizer'))
    # scheduler.load_state_dict(checkpoint.pop('scheduler'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model

def load_LMSC_model(model, filepath, logger):
  '''
  Load checkpoint file
  '''

  # check that path exists and load everything to return
  assert os.path.isfile(filepath), '=> No file found at {}'
  checkpoint = torch.load(filepath)

  if isinstance(model, (DataParallel, DistributedDataParallel)):
    model.module.load_state_dict(checkpoint.pop('model'))
  else:
    model.load_state_dict(checkpoint.pop('model'))
  logger.info('=> Model loaded at {}'.format(filepath))
  return model

def load_panoptic(model,scheduler,optimizer, path, logger):
  '''
  Load checkpoint file
  '''

  if os.path.exists(path):
    print(path)
    file_path = sorted(glob(os.path.join(path, '*.pth')))[1]
    checkpoint = torch.load(file_path, map_location='cpu')
    model = load_panoptic_model(model,checkpoint['model'])
    epoch = checkpoint.pop('startEpoch')
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    scheduler.load_state_dict(checkpoint.pop('scheduler'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model, optimizer, scheduler, epoch
  else:
    logger.info('=> No checkpoint. Initializing model from scratch')
    model.weights_init()
    epoch = 1
    return model, optimizer, scheduler, epoch

def load_panoptic_model(model,pretrained_model):
    model_dict = model.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model) 
    model.load_state_dict(model_dict)
    return model
  





def save_LMSC(path, model, optimizer, scheduler, epoch, config):
  '''
  Save checkpoint file
  '''

  # Remove recursively if epoch_last folder exists and create new one
  _remove_recursively(path)
  _create_directory(path)

  weights_fpath = os.path.join(path, 'LMSCNet_epoch_{}.pth'.format(str(epoch).zfill(3)))

  torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'config_dict': config
  }, weights_fpath)

  return weights_fpath

def save_panoptic(path, model, optimizer, scheduler, epoch):
  '''
  Save checkpoint file
  '''
  weights_fpath = os.path.join(path, 'Panoptic_epoch_{}.pth'.format(str(epoch).zfill(3)))

  torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
  }, weights_fpath)
  print('model saved to %s' % weights_fpath)
