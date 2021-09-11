from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import os
from glob import glob



def load(model, optimizer, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if os.path.exists(path):
    checkpoint = torch.load(path)
    model = load_pretrained_model(model,checkpoint['model'])
    epoch = checkpoint['startEpoch']
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
  state = {
    'startEpoch': epoch+1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()}
  torch.save(state, path)
  print('model saved to %s' % path)

def load_pretrained_model(model,pretrained_model):
    model_dict = model.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model) 
    model.load_state_dict(model_dict)
    return model