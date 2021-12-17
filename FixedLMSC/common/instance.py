import torch
from common.instance_post_processing import get_panoptic_segmentation
def get_instance(p_args,sem,center,offset,dset,train=True):
    if train:
        batch_size = p_args['model']['train_batch_size']
    else:
        batch_size = 1
    panoptic_labels=[]
    for i in range(batch_size):
        panoptic_label, _ = get_panoptic_segmentation(sem[i].unsqueeze(0), center[i].unsqueeze(0), offset[i].unsqueeze(0), dset.dataset.thing_list,\
                                                                threshold=p_args['model']['post_proc']['threshold'], nms_kernel=p_args['model']['post_proc']['nms_kernel'],\
                                                                top_k=p_args['model']['post_proc']['top_k'], polar=p_args['model']['polar'])
        panoptic_labels.append(panoptic_label)
    panoptic_labels=torch.cat(panoptic_labels,dim=0)
    print(panoptic_labels.shape)
    inst_labels = []
    instances = []
    inst_label = torch.unique(panoptic_labels)
    for things in dset.dataset.thing_list:
        inst_labels.append(inst_label[(inst_label&0xFFFF) == things])
    inst_labels = torch.cat(inst_labels,dim=0)
    print(inst_labels.shape[0])
    for instance in inst_labels:
        instances.append((panoptic_labels==instance).nonzero()[:,1:])
    print(len(instances))
    
    return instances,panoptic_labels