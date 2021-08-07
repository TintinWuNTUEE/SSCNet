from LMSCNet.models.LMSCNet import LMSCNet
from LMSCNet.models.LMSCNet_SS import LMSCNet_SS
from LMSCNet.models.SSCNet_full import SSCNet_full
from LMSCNet.models.SSCNet import SSCNet


def get_model(dataset):

  nbr_classes = dataset.nbr_classes
  grid_dimensions = dataset.grid_dimensions
  class_frequencies = dataset.class_frequencies

  model = LMSCNet_SS(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
  

  return model