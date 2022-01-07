import torch
import numpy as np

SMOOTH = 1e-6

def iou_pytorch(out: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    out = torch.argmax(out, dim=1)
    out = out.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).long()
    
    intersection = (out == labels).sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (out | labels).sum((1, 2, 3))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    print('intersection = ', intersection)
    print('union = ',union)
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou  # Or thresholded.mean() if you are interested in average across the batch

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

def iou(pred, target, n_classes = 12):
    ious = []
    # pred = pred.view(-1)
    # target = target.view(-1)
    pred = torch.argmax(pred, dim=1).unsqueeze(1)
    # pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    # target = target.squeeze(1).long() 
    target = target.long() 

    # Ignore IoU for background class ("0")
    for i in range(pred.shape[0]):
        for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred[i] == cls
            target_inds = target[i] == cls
            # print(target_inds.sum())
            intersection = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
            # print(intersection)
            union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
    return ious