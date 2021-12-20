import numpy as np
import torch
from common.instance_post_processing import get_panoptic_segmentation
def get_instance(p_args,sem,center,offset,dset,train=True,label=False):
    '''
    Get instance and return a list of instance labels
    '''
    inst_label = []
    instance = []
    panoptic_label,_ = get_panoptic_segmentation(sem, center, offset, dset.dataset.thing_list,\
                                                                threshold=p_args['model']['post_proc']['threshold'], nms_kernel=p_args['model']['post_proc']['nms_kernel'],\
                                                                top_k=p_args['model']['post_proc']['top_k'], polar=p_args['model']['polar'],label=label)
    label = torch.unique(panoptic_label)
    for things in dset.dataset.thing_list:
        inst_label.append(label[(label&0xFFFF) == things])
    inst_label = torch.cat(inst_label,dim=0)
    for inst in inst_label:
        instance.append((panoptic_label==inst).nonzero()[:,1:])  
    
    return instance,inst_label

def get_unique_label(dset):
    '''
    Get unique label from learning map
    '''
    SemKITTI_label_name = dict()
    for i in sorted(list(dset.dataset.dataset_config['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[dset.dataset.dataset_config['learning_map'][i]] = dset.dataset.dataset_config['labels'][i]
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]
    return unique_label,unique_label_str

