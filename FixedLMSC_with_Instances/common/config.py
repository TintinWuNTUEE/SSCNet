import yaml
import os
import datetime
def get_date_sting():
  '''
  To retrieve time in nice format for string printing and naming
  :return:
  '''
  _now = datetime.datetime.now()
  _date = ('%.2i' % _now.month) + ('%.2i' % _now.day) # ('%.4i' % _now.year) +
  _time = ('%.2i' % _now.hour) + ('%.2i' % _now.minute) + ('%.2i' % _now.second)
  return (_date + '_' + _time)
def merge_configs(cfgs,new_cfgs):
    if hasattr(cfgs, 'data_dir'):
        new_cfgs['dataset']['path']=cfgs.data_dir
    if hasattr(cfgs, 'model_save_path'):
        new_cfgs['model']['model_save_path']=cfgs.model_save_path
    if hasattr(cfgs, 'pretrained_model'):
        new_cfgs['model']['pretrained_model']=cfgs.pretrained_model
    return new_cfgs

class CFG:

  def __init__(self):
    '''
    Class constructor
    :param config_path:
    '''

    # Initializing dict...
    self._dict = {}
    return

  def from_config_yaml(self, config_path):
    '''
    Class constructor
    :param config_path:
    '''

    # Reading config file
    self._dict = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    self._dict['STATUS']['CONFIG'] = config_path

    if not 'OUTPUT_PATH' in self._dict['OUTPUT'].keys():
      self.set_output_filename()
      self.init_stats()
      self.update_config()

    return

  def from_dict(self, config_dict):
    '''
    Class constructor
    :param config_path:
    '''

    # Reading config file
    self._dict = config_dict
    return

  def set_output_filename(self):
    '''
    Set output path in the form Model_Dataset_DDYY_HHMMSS
    '''
    datetime = get_date_sting()
    model = self._dict['MODEL']['TYPE']
    dataset = self._dict['DATASET']['TYPE']
    OUT_PATH = os.path.join(self._dict['OUTPUT']['OUT_ROOT'], model + '_' + dataset + '_' + datetime)
    self._dict['OUTPUT']['OUTPUT_PATH'] = OUT_PATH
    return

  def update_config(self, resume=False, checkpoint_path=None):
    '''
    Save config file
    '''
    
    if resume:
      self.set_resume()
    if checkpoint_path != None:
      self._dict['STATUS']['LAST'] = checkpoint_path
    
    yaml.dump(self._dict, open(self._dict['STATUS']['CONFIG'], 'w'))
    return
  

  def init_stats(self):
    '''
    Initialize training stats (i.e. epoch mean time, best loss, best metrics)
    '''
    self._dict['OUTPUT']['BEST_LOSS'] = 999999999999
    self._dict['OUTPUT']['BEST_METRIC'] = -999999999999
    self._dict['STATUS']['LAST'] = ''
    return

  def set_resume(self):
    '''
    Update resume status dict file
    '''
    if not self._dict['STATUS']['RESUME']:
      self._dict['STATUS']['RESUME'] = True
    return

  def finish_config(self):
    self.move_config(os.path.join(self._dict['OUTPUT']['OUTPUT_PATH'], 'config.yaml'))
    return

  def move_config(self, path):
    # Remove from original path
    os.remove(self._dict['STATUS']['CONFIG'])
    # Change ['STATUS']['CONFIG'] to new path
    self._dict['STATUS']['CONFIG'] = path
    # Save to routine output folder
    yaml.dump(self._dict, open(path, 'w'))

    return
