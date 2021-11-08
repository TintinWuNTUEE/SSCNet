from glob import glob
import os
import torch
import numpy as np
from torch._C import dtype
import yaml
import matplotlib.pyplot as plt

dataset_config = yaml.safe_load(open(os.path.join('./semantic-kitti.yaml'), 'r'))

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def get_remap_lut():
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())

    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut


indice = np.moveaxis(np.array(np.meshgrid(np.arange(0,256,1),np.arange(0,256,1))),1,2)
indice = np.array(np.meshgrid(np.arange(0,256,1),np.arange(0,256,1)))
# print(indice.shape)
# print(indice[:,0,0])
# print(indice[:,0,1])
# print(indice[:,1,0])
# print(indice[:,0,2])

file_path = '../semanticKITTI/dataset/sequences'
sequence = '04'
file_type = 'preprocess'

preprocess = sorted(glob(os.path.join(file_path, sequence, file_type, "*.pt")))[::5]
label = sorted(glob(os.path.join(file_path, sequence, "voxels", "*.label")))
point = sorted(glob(os.path.join(file_path, sequence, "velodyne","*.bin")))[::5]
# print(len(preprocess))
# print(len(label))
raw_data = np.fromfile(point[3], dtype=np.float32).reshape((-1, 4))
point_mask = (raw_data[:,0]<=25.6)*(raw_data[:,0]>=-25.6)*(raw_data[:,1]<=51.2)*(raw_data[:,1]>=0)*(raw_data[:,2]>=0)*(raw_data[:,2]<=6.4)
print(point_mask.shape) 
print(raw_data.shape)
print(raw_data[:,0].max(),raw_data[:,0].min())
print(raw_data[:,1].max(),raw_data[:,1].min())
print(raw_data[:,2].max(),raw_data[:,2].min())
print(raw_data[:,3].max(),raw_data[:,3].min())
raw_data = raw_data[point_mask]
print("=====")
print(raw_data.shape)
print(raw_data[:,0].max(),raw_data[:,0].min())
print(raw_data[:,1].max(),raw_data[:,1].min())
print(raw_data[:,2].max(),raw_data[:,2].min())
print(raw_data[:,3].max(),raw_data[:,3].min())
plot3 = plt.figure(3)
plt.scatter(raw_data[:,0],raw_data[:,1])

data = list(torch.load(preprocess[3]))
voxel_label = np.fromfile(label[3], dtype=np.uint16).reshape(256,256,32)
# print(preprocess[0],label[0])
# print(voxel_label[193,212,:])
# print((voxel_label==252).sum())
# print((voxel_label==10).sum())
remap_lut = get_remap_lut()
# position = np.where((voxel_label==10))
# l = np.array(list(zip(position[0],position[1])))
# print(l)
# print(voxel_label[l[:]])

voxel_label = remap_lut[voxel_label.astype(np.uint16)]
# print(voxel_label[position[0],position[1]])

voxel_label = voxel_label.reshape(256,256,32)#[:,::-1,:]
# print(voxel_label.shape)

x = np.arange(20,dtype=int)
# print("============")
# print(data[0])
# print(type(data[0]))
# print(data[0].shape)
data[0] = np.array(data[0]) + 1
# print(x)
# print((voxel_label!=0).sum())
for i in x:
    print(i, ((data[0]==i).sum())/22219)
    print(i, ((voxel_label==i).sum())/131390)

# print(((data[1]>0.)&(data[1]<1.)).sum())
center = data[1]>0
color = np.empty(center.shape, dtype=object)
color[(data[1]>0.)&(data[1]<1.)] = 'blue'
color[data[1]==1] = 'red'
# get thing points mask
mask = np.zeros_like(voxel_label,dtype=bool)
mask2 = np.zeros_like(voxel_label,dtype=bool)

# print(mask.shape)
thing_list = [i for i in dataset_config['thing_class'].keys() if dataset_config['thing_class'][i]==True]
# print(thing_list)
for label in thing_list:
    mask[voxel_label == label] = True
    mask2[data[0]==label] = True
voxel_label[~mask] = 0
data[0][~mask2] = 0
# print(voxel_label.shape)
bev = (voxel_label>0).sum(axis=2)
bev2 = (data[0]>0).sum(axis=2)
# voxel_label = voxel_label>0
# print(voxel_label.shape)
plot1 = plt.figure(0)
plt.imshow(bev,cmap=plt.cm.gray,origin='lower')
bev_nonzero = np.nonzero(bev)
for row, col in zip(bev_nonzero[0],bev_nonzero[1]):
    for i in range(32):
        if voxel_label[row,col,i] != 0:
            plt.text( col,row, str(voxel_label[row,col,i]),color="green",fontsize=12)
            break
plot2 = plt.figure(1)
plt.imshow(bev2, cmap=plt.cm.gray,origin='lower')
bev2_nonzero = np.nonzero(bev2)
for row, col in zip(bev2_nonzero[0],bev2_nonzero[1]):
    for i in range(32):
        # print(data[0][row, col,:])
        if data[0][row, col, i] != 0:
            plt.text(col, row, str(data[0][row, col, i]),color="green",fontsize=12)
            break

# print(dataset_config['color_map'])
func = np.vectorize(dataset_config['color_map'].get)
# voxel_label = func([1,2,3])
# print("===========")
# print(voxel_label.shape)
# print(mask.shape)
# mask = np.array(np.where(mask>0))
mask = mask>0
# print(mask.shape)
# ax = plt.figure().add_subplot(projection='3d')
# center = np.expand_dims(data[1],axis=2)
color = np.empty([256,256,32,4],dtype=np.float32)
color[:,:,:] = [1,0,0,.5]
# ax.voxels(np.array([mask[0],mask[1],mask[2]]))
# NotImplementedError: Axes3D currently only supports the aspect argument 'auto'. You passed in 'equal'.
# ax.voxels(mask)
# print(data[1].squeeze().shape)

blue = (data[1]>0.).squeeze().sum()
red = (data[1]==1.).squeeze().sum()
yellow = (data[1]>1.).squeeze().sum()
# print(blue)
# print(red)
# print(yellow)

# plt.plot(blue,color='b')
# plt.plot(red,color='r')
plot1 = plt.figure(2)
position = np.where(data[1]==1)
# print(position)
plt.imshow(data[1].squeeze(), cmap='hot',origin='lower')
for row, col in zip(position[1],position[2]):
    # print(data[0][col,row,:])
    for i in range(32):
        if data[0][row,col,i] != 0:
            plt.text( col,row, str(data[0][row,col,i]),color="red",fontsize=12)
            # print("hehe")
            break



fig, ax = plt.subplots()
# a = ax.quiver(indice[0,:,:],indice[1,:,:],data[2][0,:,:],data[2][1,:,:])
a = ax.quiver(data[2][1,:,:],data[2][0,:,:],angles='xy',scale_units='xy',scale=1)

# print(data[2][0][142][173])
# print(data[2][1][142][173])

plt.show()

