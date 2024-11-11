
# Code Implementation for "Multidimensional Street View Representation and Association Analysis for Exploring Human Subjective Perception Differences in Cross-Cultural Urban Landscapes"

This repository provides the source code implementation associated with the manuscript titled **"Multidimensional Street View Representation and Association Analysis for Exploring Human Subjective Perception Differences in Cross-Cultural Urban Landscapes"**. It offers tools for analyzing the similarity and heterogeneity of urban street views across different cultural areas.

### Project Structure and File Descriptions

0. **`pd_valid_sample_sets.csv`**: A filtered dataset of valid street view images from various regions. The valid samples filter is implemented by scores.py.
1. **`DeepLabV3Plus-Pytorch`**: Implementation of the semantic segmentation model for street view images. The `svi_seg.py` script utilizes the DeepLabV3Plus model, pre-trained on the Cityscapes dataset, to perform semantic segmentation of 19 urban landscape object classes.
2. **`Pytorch-Mask-RCNN`**: Implementation of the instance segmentation model for street view images. The `instance_seg.py` script is based on the Mask-RCNN architecture and enables counting of vehicles and pedestrians within street view images.
3. **`index_calcu.py`**: Script for computing visual representation vectors for each image.
4. **`vectors.py`**: Integrates the calculation of physical and visual representation vectors for street view images by calling the results from the previous three steps.
5. **`regression.py`**: Implements the logistic regression model.

### 1. Basic Requirements

The code has been tested on **Windows 11** with an **RTX 4070TI 12GB** GPU. Since a pre-trained backbone and a relatively small dataset are used, there is no high demand on GPU resources.

**Key Environment and Version Requirements**
```shell
CUDA11
pip install pytorch == 1.11.0
torchvision == 0.12.0
```

## 2. Execution

Run the following scripts in sequence to reproduce the analysis:

```python
# Perform semantic segmentation
python svi_seg.py

# Perform instance segmentation
python instance_seg.py

# Compute scene representation vectors for all street view images
python vectors.py

# Conduct regression analysis to correlate human perception of streetscape characteristics with their environmental features
python regression.py
```

### Additional Resources

**PP2.0 Dataset**  
[Place Pulse 2.0](https://pan.baidu.com/s/1fYoVE6YKcbDJiVqoxx5K9g#list/path=%2F)

**Model Weights**  
- [DeepLabV3+ Pre-trained Weights](https://share.weiyun.com/qqx78Pv5)
- [MaskRCNN Pre-trained Weights](https://pan.baidu.com/s/1n97fyrqYAOIS7dIB_NXgHw) (Access Code: dk8s)
