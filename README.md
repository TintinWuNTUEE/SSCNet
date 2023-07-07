# SSCNET![](https://github.com/TintinWuNTUEE/SSCNet/blob/main/intro.png)


## Method
Inspired by [SISNet](https://arxiv.org/abs/2104.03640) on indoor scenes tasks, we present a scene-instance-scene approach on outdoor scene datasets.
### 1. Dataset
[SemanticKITTI](http://www.semantic-kitti.org/)
Preprocessed with KNN on Panoptic Segmentation Task and Semantic Scene Completion Task for the label of the foreground instances
### 2. Scene Completion
Use [LMSCNet](https://arxiv.org/abs/2008.10559) as backbone, roughly complete the scene without taking too much computation resource.
### 3. Panoptic Segmentation
For instance proposal, we decided to use [Panoptic Polarnet](https://arxiv.org/abs/2103.14962) for foreground instance proposal since the dataset doesn't include bounding boxes as normal pointcloud datasets.
### 4.Instance Completion
For instance completion we decided to use [PCN](https://arxiv.org/abs/1808.00671) with point cloud padding since each instances has different number of points.

### 5.Instance Refinement
Project the instances with completed instances back to the scene and perform Scene Completion again for final refinement