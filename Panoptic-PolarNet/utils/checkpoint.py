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
    modelDict = model.state_dict()
    pretrainedStateDict= checkpoint['model']
    pretrained_model = {k: v for k, v in pretrainedStateDict.items() if k in pretrainedStateDict}
    modelDict.update(pretrained_model) 
    epoch = checkpoint['startEpoch']
    model.load_state_dict(modelDict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(path))
    return model, optimizer,  epoch
  else:
    logger.info('=> No checkpoint. Initializing model from scratch')
    epoch = 1
    return model, optimizer,  epoch


def save(path, model, optimizer, epoch):
  '''
  Save checkpoint file
  '''
  state = {'startEpoch': epoch+1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()}
  torch.save(state, path)
  print('model saved to %s' % path)