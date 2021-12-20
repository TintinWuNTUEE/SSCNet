import numpy as np
import torch
from common.instance_post_processing import get_panoptic_segmentation
def normalize(instance,center):
    print(center.shape)
    xavg,yavg,zavg = center[1],center[0],torch.mean(instance[:,2])
    torch.sub(instance,torch.tensor((xavg,yavg,zavg)))
    instance=torch.nn.functional.normalize(instance)
    return instance
def get_instance(p_args,sem,center,offset,dset,train=True):
    if train:
        batch_size = p_args['model']['train_batch_size']
    else:
        batch_size = 1
    panoptic_labels=[]
    panoptic_centers = []
    for i in range(batch_size):
        # print(center.shape)
        panoptic_label, center = get_panoptic_segmentation(sem[i].unsqueeze(0), center[i].unsqueeze(0), offset[i].unsqueeze(0), dset.dataset.thing_list,\
                                                                threshold=p_args['model']['post_proc']['threshold'], nms_kernel=p_args['model']['post_proc']['nms_kernel'],\
                                                                top_k=p_args['model']['post_proc']['top_k'], polar=p_args['model']['polar'])
        panoptic_labels.append(panoptic_label)
        panoptic_centers.append(center)
    panoptic_labels=torch.cat(panoptic_labels,dim=0)
    panoptic_centers = torch.cat(panoptic_centers,dim=0)
    # print(panoptic_labels.shape)
    inst_labels = []
    instances = []
    centers=[]
    inst_label = torch.unique(panoptic_labels)
    for things in dset.dataset.thing_list:
        inst_labels.append(inst_label[(inst_label&0xFFFF) == things])
    inst_labels = torch.cat(inst_labels,dim=0)
    for instance in inst_labels:
        instances.append((panoptic_labels==instance).nonzero()[:,1:].type(torch.FloatTensor))
    # for i in range(len(instances)):
    #     instances[i]=normalize(instances[i],panoptic_centers[i])
    return instances,panoptic_labels
def get_unique_label(dset):
    SemKITTI_label_name = dict()
    for i in sorted(list(dset.dataset.dataset_config['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[dset.dataset.dataset_config['learning_map'][i]] = dset.dataset.dataset_config['labels'][i]
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]
    return unique_label,unique_label_str