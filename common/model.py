# from SSCNet.models.LMSCNet import LMSCNet
from SSCNet.models.LMSCNet_SS import LMSCNet_SS
# from SSCNet.models.SSCNet_full import SSCNet_full
# from SSCNet.models.SSCNet import SSCNet


def get_model(_cfg, dataset):

  nbr_classes = dataset.nbr_classes
  grid_dimensions = dataset.grid_dimensions
  class_frequencies = dataset.class_frequencies

  model = LMSCNet_SS(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
  

  return model