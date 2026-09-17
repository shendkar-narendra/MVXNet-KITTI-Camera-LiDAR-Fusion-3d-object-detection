## ADD BEV VIEW for all three models

## If your PointPillars model achieved higher accuracy than MVX-Net, it usually indicates an issue with data alignment, sensor calibration, or model tuning --- this means our mvx net is mostly wrong or we need proper justification 

<img width="1242" height="375" alt="004489_mvxnet" src="https://github.com/user-attachments/assets/8244af82-ee37-41dc-9f3a-51c3642ecd22" />
<img width="1242" height="375" alt="003705_mvxnet" src="https://github.com/user-attachments/assets/4107a472-f342-452a-915d-a0d5a7a87168" />


# MVXNet-KITTI-Camera-LiDAR-Fusion

Camera-LiDAR sensor-fusion-based 3D object detection on the **KITTI 3D Object Detection Dataset** using **Dynamic MVX-Net / DynamicMVXFasterRCNN** implemented with **MMDetection3D 0.17.1**.

This repository contains the experiment configuration, training information, evaluation workflow, prediction results, visualization code, and reproducibility instructions used for the **camera + LiDAR fusion** part of a Master's scientific project comparing different 3D perception approaches.

The detector was trained to detect the three standard KITTI road-user classes:

- **Car**
- **Pedestrian**
- **Cyclist**

The final experiment was trained for **80 epochs** using both RGB camera images and LiDAR point clouds.

---

# 1. Project Overview

Modern autonomous-driving perception systems can use several sensor modalities.

Camera sensors provide:

- color
- texture
- semantic information
- high-resolution appearance information

LiDAR sensors provide:

- accurate 3D geometry
- depth
- distance information
- spatial structure

This experiment investigates a **multi-modal fusion approach** in which camera features and LiDAR features are combined before performing 3D object detection.

The model used in this experiment is based on the MVX-Net family implemented in MMDetection3D as:

```python
DynamicMVXFasterRCNN
```

The experiment was performed using:

```text
Framework: MMDetection3D
MMDetection3D version: 0.17.1
Dataset: KITTI 3D Object Detection
Modalities: Camera + LiDAR
Classes: Pedestrian, Cyclist, Car
Training epochs: 80
```

---

# 2. Repository Structure

Recommended repository structure:

```text
MVXNet-KITTI-Camera-LiDAR-Fusion/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/
│   └── dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
│
├── scripts/
│   └── visualize_mvxnet.py
│
├── results/
│   ├── mvxnet_epoch80_results.pkl
│   └── training_log.log
│
├── figures/
│   └── predictions/
│       ├── prediction_01.png
│       ├── prediction_02.png
│       ├── prediction_03.png
│       └── ...
│
├── weights/
│   └── README.md
│
└── docs/
    └── environment.md
```

The original MMDetection3D project used during development was located at:

```text
D:\mvxnet-mmdetection3d
```

The important original files were:

```text
Main config:
D:\mvxnet-mmdetection3d\configs\mvxnet\
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py

Training script:
D:\mvxnet-mmdetection3d\tools\train.py

Testing script:
D:\mvxnet-mmdetection3d\tools\test.py

KITTI dataset:
D:\mvxnet-mmdetection3d\data\kitti\

Training checkpoints:
D:\mvxnet-mmdetection3d\work_dirs\
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class\

Final prediction file:
D:\mvxnet-mmdetection3d\work_dirs\
mvxnet_epoch80_results.pkl

Custom visualization script:
D:\mvxnet-mmdetection3d\visualize_mvxnet.py
```

---

# 3. Model Architecture

The model uses two sensor branches:

1. Camera branch
2. LiDAR branch

The extracted features are combined through a point-image fusion module.

A simplified architecture is:

```text
                  CAMERA IMAGE
                       │
                       ▼
                  ResNet-50
                       │
                       ▼
                      FPN
                       │
                       │
                       ├─────────────────────┐
                       │                     │
                       │                     ▼
LiDAR Point Cloud      │                Image Features
       │               │                     │
       ▼               │                     │
Dynamic Voxelization   │                     │
       │               │                     │
       ▼               │                     │
   DynamicVFE ◄────────┴─────────────────────┘
       │
       ▼
   PointFusion
       │
       ▼
 SparseEncoder
       │
       ▼
     SECOND
       │
       ▼
   SECONDFPN
       │
       ▼
 Anchor3DHead
       │
       ▼
3D Bounding Box Predictions
       │
       ├── Pedestrian
       ├── Cyclist
       └── Car
```

---

# 4. Main Model Components

## 4.1 Camera Backbone — ResNet-50

The image branch uses:

```python
type='ResNet'
depth=50
```

ResNet-50 extracts hierarchical features from the RGB camera image.

The backbone produces feature maps at multiple resolutions.

---

## 4.2 Feature Pyramid Network — FPN

The image backbone is followed by an FPN:

```python
type='FPN'

in_channels=[
    256,
    512,
    1024,
    2048
]

out_channels=256
num_outs=5
```

The Feature Pyramid Network combines image features from different spatial resolutions.

This allows the detector to use both:

- low-level spatial information
- high-level semantic information

---

# 5. LiDAR Processing

## 5.1 Point Cloud Input

The LiDAR point cloud contains four values per point:

```text
x
y
z
reflectance/intensity
```

The configuration loads the point cloud using:

```python
dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=4
)
```

---

# 6. Point Cloud Range

Only LiDAR points inside the following 3D region are processed:

```python
point_cloud_range = [
    0,
    -40,
    -3,
    70.4,
    40,
    1
]
```

Meaning approximately:

```text
X: 0 to 70.4 meters
Y: -40 to +40 meters
Z: -3 to +1 meters
```

This defines the region in front of the vehicle used by the detector.

---

# 7. Dynamic Voxelization

The point cloud is divided into 3D voxels.

The voxel size used was:

```python
voxel_size = [
    0.1,
    0.1,
    0.1
]
```

Therefore each voxel corresponds approximately to:

```text
0.1 m × 0.1 m × 0.1 m
```

Unlike fixed voxelization, the configuration uses dynamic voxel processing:

```python
pts_voxel_encoder=dict(
    type='DynamicVFE'
)
```

---

# 8. Dynamic Voxel Feature Encoder

The LiDAR branch uses:

```python
DynamicVFE
```

with:

```python
in_channels=4
feat_channels=[64, 64]
```

The VFE transforms raw LiDAR point features into learned feature representations.

Additional geometric information is included through:

```python
with_cluster_center=True
with_voxel_center=True
```

These features help the network understand the spatial relationship between points and voxels.

---

# 9. Camera-LiDAR Fusion

The key multi-modal component is:

```python
PointFusion
```

The configuration uses:

```python
fusion_layer=dict(
    type='PointFusion',
    img_channels=256,
    pts_channels=64,
    mid_channels=128,
    out_channels=128,
    img_levels=[0, 1, 2, 3, 4],
    align_corners=False,
    activate_out=True,
    fuse_out=False
)
```

PointFusion associates LiDAR points with image features obtained from the camera branch.

Conceptually:

```text
3D LiDAR point
       │
       │ Projection using camera calibration
       ▼
Corresponding image location
       │
       ▼
Image feature sampled from FPN
       │
       ▼
Camera feature + LiDAR feature
       │
       ▼
Fused feature representation
```

The fused representation contains information from both sensors.

---

# 10. Sparse 3D Encoder

After feature fusion, the model uses:

```python
type='SparseEncoder'
```

with:

```python
in_channels=128
sparse_shape=[41, 800, 704]
```

Sparse convolution is useful for LiDAR because most of the 3D space contains no points.

Rather than processing every possible voxel, sparse convolution primarily processes occupied regions.

This greatly reduces unnecessary computation.

---

# 11. SECOND Backbone

The encoded LiDAR representation is processed using the SECOND backbone:

```python
type='SECOND'

in_channels=256

layer_nums=[
    5,
    5
]

layer_strides=[
    1,
    2
]

out_channels=[
    128,
    256
]
```

SECOND is a common backbone for LiDAR-based 3D object detection.

---

# 12. SECOND Feature Pyramid Network

The LiDAR backbone output is passed to:

```python
SECONDFPN
```

Configuration:

```python
in_channels=[
    128,
    256
]

upsample_strides=[
    1,
    2
]

out_channels=[
    256,
    256
]
```

The output features are combined and passed to the 3D detection head.

---

# 13. 3D Detection Head

The model uses:

```python
Anchor3DHead
```

with:

```python
num_classes=3
```

The three object classes are:

```python
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car'
]
```

The detector predicts:

- object class
- 3D position
- width
- length
- height
- orientation
- confidence score

---

# 14. Anchor Sizes

Different anchor dimensions are used for different KITTI object classes.

The configuration contains:

```python
sizes=[
    [0.6, 0.8, 1.73],
    [0.6, 1.76, 1.73],
    [1.6, 3.9, 1.56]
]
```

These correspond approximately to typical dimensions of:

```text
Pedestrian
Cyclist
Car
```

Two anchor orientations are used:

```python
rotations=[
    0,
    1.57
]
```

where:

```text
1.57 rad ≈ 90 degrees
```

---

# 15. Dataset

The experiment uses the:

```text
KITTI 3D Object Detection Dataset
```

The training split was prepared inside:

```text
data/kitti/
```

The main information files used were:

```text
data/kitti/kitti_infos_train.pkl
data/kitti/kitti_infos_val.pkl
```

The experiment uses the KITTI training data split into:

```text
training subset
validation subset
```

The official KITTI test set was not required for the internal experiment evaluation.

The validation information file contains:

```text
3769 validation samples
```

---

# 16. Expected KITTI Directory Structure

After downloading and preparing KITTI, the structure should approximately look like:

```text
data/
└── kitti/
    │
    ├── training/
    │   ├── calib/
    │   ├── image_2/
    │   ├── label_2/
    │   ├── velodyne/
    │   └── velodyne_reduced/
    │
    ├── testing/
    │   ├── calib/
    │   ├── image_2/
    │   └── velodyne/
    │
    ├── ImageSets/
    │   ├── train.txt
    │   ├── val.txt
    │   └── ...
    │
    ├── kitti_infos_train.pkl
    └── kitti_infos_val.pkl
```

> The KITTI dataset itself is **not included in this repository**.

Users must obtain KITTI separately and prepare it according to MMDetection3D requirements.

---

# 17. Dataset Modalities

Both sensors are enabled:

```python
input_modality=dict(
    use_lidar=True,
    use_camera=True
)
```

This is what makes the experiment a multi-modal sensor fusion experiment.

---

# 18. Image Preprocessing

Images are resized using:

```python
img_scale=(1280, 384)
```

Image normalization uses:

```python
mean=[
    103.53,
    116.28,
    123.675
]

std=[
    1.0,
    1.0,
    1.0
]

to_rgb=False
```

Images are also padded:

```python
dict(
    type='Pad',
    size_divisor=32
)
```

---

# 19. Training Data Augmentation

Several augmentation techniques are applied during training.

## Global Rotation

```python
rot_range=[
    -0.78539816,
    0.78539816
]
```

Approximately:

```text
-45° to +45°
```

---

## Scale Augmentation

```python
scale_ratio_range=[
    0.95,
    1.05
]
```

---

## Translation Augmentation

```python
translation_std=[
    0.2,
    0.2,
    0.2
]
```

---

## Horizontal Flip

```python
flip_ratio_bev_horizontal=0.5
```

This means a 50% probability of horizontal flipping during training.

---

## Point Shuffling

```python
dict(
    type='PointShuffle'
)
```

The order of point-cloud points is randomized.

---

# 20. Training Dataset Repetition

The training dataset is wrapped using:

```python
type='RepeatDataset'
times=2
```

This means the training dataset is repeated twice for each logical training epoch.

---

# 21. Training Configuration

The final training configuration used:

```python
runner=dict(
    type='EpochBasedRunner',
    max_epochs=80
)
```

Therefore:

```text
Total training target = 80 epochs
```

---

# 22. Optimizer

The optimizer used was:

```python
AdamW
```

Configuration:

```python
optimizer=dict(
    type='AdamW',
    lr=0.003,
    betas=(0.95, 0.99),
    weight_decay=0.01
)
```

---

# 23. Learning Rate

Initial learning rate:

```text
0.003
```

The learning-rate policy was:

```python
CosineAnnealing
```

Configuration:

```python
lr_config=dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05
)
```

The learning rate therefore gradually decreases during training according to a cosine schedule.

---

# 24. Learning Rate Warmup

The first:

```text
1000 iterations
```

use linear learning-rate warmup.

The warmup ratio is:

```python
0.1
```

Warmup helps avoid unstable parameter updates at the beginning of training.

---

# 25. Gradient Clipping

Gradient clipping was enabled:

```python
optimizer_config=dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2
    )
)
```

Gradient clipping prevents extremely large gradient values from destabilizing training.

---

# 26. Loss Functions

The model uses several losses.

## Classification Loss

```python
FocalLoss
```

Configuration:

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0
)
```

Focal Loss reduces the influence of easy background samples and places greater emphasis on difficult examples.

---

## Bounding Box Regression Loss

```python
SmoothL1Loss
```

Configuration:

```python
loss_bbox=dict(
    type='SmoothL1Loss',
    beta=0.1111111111111111,
    loss_weight=2.0
)
```

This loss trains the model to predict accurate 3D bounding-box parameters.

---

## Direction Classification Loss

```python
CrossEntropyLoss
```

Configuration:

```python
loss_dir=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=0.2
)
```

The direction loss helps determine the orientation of detected objects.

---

# 27. Batch Size

The final logged experiment used:

```python
samples_per_gpu=1
```

Therefore:

```text
Batch size per GPU = 1
```

This small batch size was appropriate for the available GPU memory.

---

# 28. Data Loader Workers

The final logged experiment used:

```python
workers_per_gpu=0
```

This setting was used for the Windows training environment.

---

# 29. Hardware Used

The experiment was performed using:

```text
GPU: NVIDIA GeForce GTX 1650
```

The training log reported GPU memory usage around:

```text
~1.4 GB during the logged training iterations
```

Actual memory usage can vary between iterations and environments.

---

# 30. Software Environment

The recorded environment was:

```text
Operating System: Windows
Platform: win32

Python:
3.8.20

GPU:
NVIDIA GeForce GTX 1650

CUDA Runtime:
11.1

PyTorch:
1.8.1+cu111

TorchVision:
0.9.1+cu111

OpenCV:
4.5.5

MMCV:
1.4.0

MMDetection:
2.14.0

MMSegmentation:
0.14.1

MMDetection3D:
0.17.1
```

The MMDetection3D revision recorded by the experiment was:

```text
0.17.1+f110797
```

---

# 31. Training Script

Training uses the standard MMDetection3D training entry point:

```text
tools/train.py
```

Original location:

```text
D:\mvxnet-mmdetection3d\tools\train.py
```

The training script itself was part of MMDetection3D and is therefore not treated as custom project code.

---

# 32. Main Configuration File

The main experiment configuration is:

```text
configs/mvxnet/
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
```

In this repository it can be stored as:

```text
configs/
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
```

This configuration controls:

- model architecture
- dataset
- classes
- preprocessing
- augmentation
- optimizer
- learning rate
- epochs
- loss functions
- evaluation
- post-processing
- checkpoints

---

# 33. Starting Training

From the root MMDetection3D directory:

```bash
cd D:\mvxnet-mmdetection3d
```

Activate the correct environment first.

Example:

```bash
conda activate mvxnet
```

Then start training with:

```bash
python tools/train.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
```

MMDetection3D automatically:

1. loads the configuration
2. creates the dataset
3. builds the model
4. initializes the optimizer
5. performs forward propagation
6. calculates losses
7. performs backpropagation
8. updates model weights
9. saves checkpoints
10. evaluates according to the configured evaluation interval

---

# 34. Training Output Directory

Training outputs were saved to:

```text
work_dirs/
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/
```

Example checkpoint files:

```text
epoch_59.pth
epoch_60.pth
epoch_61.pth
...
epoch_78.pth
epoch_79.pth
epoch_80.pth
latest.pth
```

The final model used for the experiment is:

```text
epoch_80.pth
```

---

# 35. Checkpoint Saving

The experiment used:

```python
checkpoint_config=dict(
    interval=1
)
```

Therefore a checkpoint was saved after every epoch.

---

# 36. `epoch_80.pth` vs `latest.pth`

At the end of training:

```text
epoch_80.pth
```

represents the explicitly named checkpoint after epoch 80.

```text
latest.pth
```

is MMDetection3D's latest-checkpoint reference.

For reproducible evaluation, this project uses:

```text
epoch_80.pth
```

because its training stage is explicit.

---

# 37. Resuming Training

The experiment was interrupted/resumed during training.

The recorded resumed run loaded:

```text
latest.pth
```

from:

```text
work_dirs/
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/
latest.pth
```

The log confirms that training resumed from:

```text
epoch 40
```

and continued toward:

```text
epoch 80
```

A resume command can be written as:

```bash
python tools/train.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py --resume-from work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/latest.pth
```

Conceptually:

```text
Epoch 1
   │
   ▼
...
   │
Epoch 40
   │
   ├── Training stopped/interrupted
   │
   ▼
latest.pth
   │
   ▼
Resume training
   │
   ▼
Epoch 41
   │
   ▼
...
   │
   ▼
Epoch 80
   │
   ▼
epoch_80.pth
```

A proper resume restores more than just model parameters.

It can restore:

- model weights
- optimizer state
- epoch number
- training state
- learning-rate scheduling state

This is why resuming differs from merely loading pretrained weights.

---

# 38. Training Loss

During training the log reports:

```text
loss_cls
loss_bbox
loss_dir
loss
grad_norm
```

Where:

### `loss_cls`

Classification loss.

Measures how well the model predicts the correct object class.

---

### `loss_bbox`

3D bounding-box regression loss.

Measures errors in predicted:

- position
- dimensions
- orientation-related box parameters

---

### `loss_dir`

Direction classification loss.

Helps determine object heading.

---

### `loss`

Overall weighted loss.

Conceptually:

```text
Total Loss
   =
Classification Loss
   +
Bounding Box Loss
   +
Direction Loss
```

with the exact contribution controlled by each loss weight.

---

# 39. Backpropagation

During neural-network training the basic process is:

```python
loss.backward()
optimizer.step()
```

### `loss.backward()`

Computes gradients of the loss with respect to the trainable network parameters.

Conceptually:

```text
Prediction
   │
   ▼
Loss
   │
   ▼
Backpropagation
   │
   ▼
Gradients
```

### `optimizer.step()`

Uses the calculated gradients to update the model parameters.

Conceptually:

```text
Current weights
      │
      ▼
Gradients
      │
      ▼
Optimizer
      │
      ▼
Updated weights
```

This process repeats for every training iteration.

---

# 40. Validation

Validation does not directly train the model.

The validation set is used to evaluate the current checkpoint.

The validation configuration uses:

```text
data/kitti/kitti_infos_val.pkl
```

Validation performs:

```text
Model
   │
   ▼
Validation Samples
   │
   ▼
Predictions
   │
   ▼
KITTI Evaluation
   │
   ▼
AP Metrics
```

No optimizer update is performed during normal validation.

Therefore validation itself does not update the learned model weights.

---

# 41. KITTI Evaluation Metrics

KITTI evaluates detection at three difficulty levels:

```text
Easy
Moderate
Hard
```

Difficulty depends on properties such as:

- bounding-box height
- occlusion
- truncation

The main evaluated representations include:

```text
2D Bounding Box AP
BEV AP
3D AP
AOS
```

---

# 42. 2D Bounding Box AP

This measures detection performance in image space.

```text
Camera image
      │
      ▼
2D rectangle
```

---

# 43. BEV AP

BEV means:

```text
Bird's-Eye View
```

The 3D box is projected onto the ground plane and evaluated from above.

```text
            Front
              ↑

        ┌───────────┐
        │    Car    │
        └───────────┘

              Ego
```

---

# 44. 3D AP

3D Average Precision evaluates overlap between the complete predicted and ground-truth 3D boxes.

This is one of the most important metrics for autonomous-driving 3D object detection.

---

# 45. AOS

AOS means:

```text
Average Orientation Similarity
```

It evaluates whether detected objects also have the correct orientation.

---

# 46. IoU Thresholds

KITTI commonly uses stricter overlap criteria for cars than for pedestrians and cyclists.

The logged experiment reports results such as:

```text
Car:
AP@0.70,0.70,0.70

Pedestrian:
AP@0.50,0.50,0.50

Cyclist:
AP@0.50,0.50,0.50
```

Additional looser-threshold results are also reported by the evaluation code.

---

# 47. Example Logged Evaluation Result

An intermediate validation result recorded during the resumed experiment at epoch 41 included:

## Pedestrian — strict 3D AP

```text
Easy:     10.0000
Moderate: 10.0566
Hard:      9.9546
```

## Cyclist — strict 3D AP

```text
Easy:     15.4985
Moderate: 11.6546
Hard:     10.7127
```

## Car — strict 3D AP

```text
Easy:     57.0238
Moderate: 47.7383
Hard:     43.7628
```

These values are shown here as an example of the KITTI evaluation produced during training.

They are **not presented as the final epoch-80 metrics unless separately confirmed from the final epoch-80 evaluation output**.

---

# 48. Testing

Testing uses the standard MMDetection3D script:

```text
tools/test.py
```

Original location:

```text
D:\mvxnet-mmdetection3d\tools\test.py
```

The basic structure is:

```bash
python tools/test.py CONFIG CHECKPOINT
```

For the final checkpoint:

```bash
python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/epoch_80.pth
```

---

# 49. Saving Test Predictions

The prediction output used in this project is:

```text
mvxnet_epoch80_results.pkl
```

A standard MMDetection3D test invocation for saving predictions uses an output argument such as:

```bash
python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/epoch_80.pth --out work_dirs/mvxnet_epoch80_results.pkl
```

This produces a serialized Python prediction file:

```text
work_dirs/mvxnet_epoch80_results.pkl
```

> The exact original shell command was not preserved in the training log included with this repository. The command above represents the standard MMDetection3D workflow corresponding to the resulting `.pkl` file.

---

# 50. Evaluating the Model

Evaluation can be performed using MMDetection3D's KITTI evaluation through the configured dataset.

A typical workflow is:

```bash
python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/epoch_80.pth --eval bbox
```

Depending on the exact MMDetection3D command-line setup, evaluation can also be performed while simultaneously saving the output predictions.

---

# 51. Prediction File

Final experiment predictions are stored in:

```text
results/mvxnet_epoch80_results.pkl
```

Original location:

```text
D:\mvxnet-mmdetection3d\work_dirs\
mvxnet_epoch80_results.pkl
```

The pickle file contains inference outputs for the configured KITTI validation dataset.

It can be reused for:

- visualization
- prediction analysis
- confidence analysis
- class analysis
- custom evaluation
- qualitative comparison

without rerunning the complete network inference every time.

---

# 52. Test Configuration

The logged inference configuration uses:

```python
score_thr=0.1
nms_thr=0.01
nms_pre=100
max_num=50
use_rotate_nms=True
```

---

# 53. Confidence Threshold

The detector configuration uses:

```python
score_thr=0.1
```

Predictions below the configured threshold can be removed during detector post-processing.

---

# 54. Non-Maximum Suppression

The configuration uses rotated NMS:

```python
use_rotate_nms=True
```

with:

```python
nms_thr=0.01
```

Non-Maximum Suppression removes multiple overlapping predictions corresponding to the same physical object.

Conceptually:

```text
Several overlapping predicted boxes
                │
                ▼
        Compare confidence
                │
                ▼
             NMS
                │
                ▼
Keep strongest prediction
```

---

# 55. Maximum Predictions

The configuration uses:

```python
max_num=50
```

Therefore the detector retains at most the configured maximum number of detections per sample after post-processing.

---

# 56. Custom Visualization

A custom visualization script was created:

```text
visualize_mvxnet.py
```

Original location:

```text
D:\mvxnet-mmdetection3d\visualize_mvxnet.py
```

Repository location:

```text
scripts/visualize_mvxnet.py
```

The script uses:

```text
Prediction file:
results/mvxnet_epoch80_results.pkl

KITTI info:
data/kitti/kitti_infos_val.pkl

Camera images:
data/kitti/training/image_2/

Calibration:
data/kitti/training/calib/
```

Visualization output is saved into a folder such as:

```text
work_dirs/mvxnet_visualization/
```

Selected examples can be included in this repository under:

```text
figures/predictions/
```

---

# 57. Visualization Threshold

The custom visualization script used a separate drawing threshold:

```python
SCORE_THRESHOLD = 0.30
```

This should not be confused with the model's internal test threshold:

```python
score_thr = 0.10
```

Therefore:

```text
Model test threshold       = 0.10
Visualization threshold    = 0.30
```

A prediction may therefore exist in the model output but not be drawn if its confidence is:

```text
0.10 <= score < 0.30
```

This is especially important when inspecting classes such as Pedestrian and Cyclist, which may have lower confidence scores than some Car predictions.

---

# 58. Running the Visualization

From the project root:

```bash
python scripts/visualize_mvxnet.py
```

If using the original MMDetection3D project layout:

```bash
python visualize_mvxnet.py
```

Make sure the paths inside the script match the local dataset and prediction paths.

---

# 59. Why Camera + LiDAR Fusion?

Camera-only systems provide strong semantic information but do not directly measure metric depth.

LiDAR-only systems provide accurate geometry and distance information but contain less visual semantic information.

Fusion aims to combine their complementary strengths:

```text
CAMERA
 ├── color
 ├── texture
 ├── semantics
 └── appearance

LiDAR
 ├── depth
 ├── 3D position
 ├── geometry
 └── distance

        │
        ▼

      FUSION

        │
        ▼

Better combined 3D representation
```

---

# 60. Why Dynamic MVX-Net?

The model allows image features to enrich point-cloud representations before final 3D detection.

Instead of independently detecting objects in each modality and combining final decisions, features from the two sensors are integrated inside the neural-network pipeline.

This makes the approach a feature-level multi-modal fusion method.

---

# 61. Experimental Workflow

The complete experimental pipeline was:

```text
1. Download KITTI dataset
             │
             ▼
2. Prepare KITTI directory
             │
             ▼
3. Generate training/validation info files
             │
             ▼
4. Configure Dynamic MVX-Net
             │
             ▼
5. Load camera + LiDAR data
             │
             ▼
6. Train model
             │
             ▼
7. Save checkpoints
             │
             ▼
8. Resume training when necessary
             │
             ▼
9. Reach epoch 80
             │
             ▼
10. Select epoch_80.pth
             │
             ▼
11. Run test.py
             │
             ▼
12. Generate mvxnet_epoch80_results.pkl
             │
             ▼
13. Calculate KITTI metrics
             │
             ▼
14. Run visualize_mvxnet.py
             │
             ▼
15. Produce qualitative prediction images
```

---

# 62. Reproducing the Experiment

## Step 1 — Clone / obtain MMDetection3D 0.17.1

Use a version compatible with the environment documented in this repository.

The original experiment used:

```text
MMDetection3D 0.17.1
MMDetection 2.14.0
MMCV 1.4.0
PyTorch 1.8.1+cu111
```

---

## Step 2 — Create the environment

An example Conda environment:

```bash
conda create -n mvxnet python=3.8
conda activate mvxnet
```

Install the versions required by the experiment.

The exact environment used is documented in:

```text
docs/environment.md
```

---

## Step 3 — Prepare KITTI

Place KITTI under:

```text
data/kitti/
```

Do not commit the KITTI dataset to GitHub.

A standard MMDetection3D KITTI preparation workflow may use the framework's data-preparation utility, for example:

```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

The required final files for this experiment include:

```text
data/kitti/kitti_infos_train.pkl
data/kitti/kitti_infos_val.pkl
```

and:

```text
data/kitti/training/image_2/
data/kitti/training/calib/
data/kitti/training/velodyne_reduced/
```

---

## Step 4 — Copy the experiment configuration

Place:

```text
dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
```

inside the appropriate MMDetection3D configuration directory:

```text
configs/mvxnet/
```

---

## Step 5 — Train

Run:

```bash
python tools/train.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
```

---

## Step 6 — Resume if interrupted

Run:

```bash
python tools/train.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py --resume-from work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/latest.pth
```

---

## Step 7 — Use final checkpoint

After training:

```text
work_dirs/
└── dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/
    └── epoch_80.pth
```

---

## Step 8 — Run inference

```bash
python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/epoch_80.pth --out work_dirs/mvxnet_epoch80_results.pkl
```

---

## Step 9 — Evaluate

Example:

```bash
python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py work_dirs/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/epoch_80.pth --eval bbox
```

---

## Step 10 — Visualize predictions

```bash
python scripts/visualize_mvxnet.py
```

---

# 63. Important Experiment Files

| File | Purpose |
|---|---|
| `dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py` | Main training and evaluation configuration |
| `tools/train.py` | Standard MMDetection3D training entry point |
| `tools/test.py` | Standard MMDetection3D testing/evaluation entry point |
| `epoch_80.pth` | Final trained checkpoint |
| `mvxnet_epoch80_results.pkl` | Final saved prediction results |
| `visualize_mvxnet.py` | Custom qualitative visualization |
| `kitti_infos_train.pkl` | Training-set metadata |
| `kitti_infos_val.pkl` | Validation-set metadata |
| `training_log.log` | Training environment, losses, evaluation and experiment record |

---

# 64. Files Included in This Repository

The repository intentionally contains only experiment-specific files.

Included:

```text
✓ Experiment configuration
✓ Custom visualization script
✓ Training log
✓ Prediction results
✓ Selected visualization examples
✓ Environment information
✓ Reproduction instructions
```

---

# 65. Files Not Included

The following are intentionally excluded.

## KITTI Dataset

```text
data/kitti/
```

The KITTI dataset must be downloaded separately.

---

## All Training Checkpoints

Intermediate checkpoints such as:

```text
epoch_1.pth
epoch_2.pth
...
epoch_79.pth
```

are not stored in the repository because each checkpoint is large.

---

## MMDetection3D Source Code

The complete upstream framework is not duplicated in this repository.

Examples not uploaded as custom code:

```text
mmdet3d/
tools/train.py
tools/test.py
tests/
docs/
docker/
```

unless a file was specifically modified for this experiment.

---

# 66. Model Weights

The final checkpoint is:

```text
epoch_80.pth
```

Its size is approximately:

```text
~395 MB
```

This is larger than GitHub's normal per-file limit.

Therefore the checkpoint should be distributed using:

```text
GitHub Releases
```

or:

```text
Git LFS
```

After downloading, place it at:

```text
weights/epoch_80.pth
```

---

# 67. `.gitignore`

Recommended `.gitignore`:

```gitignore
# KITTI dataset
data/

# Model checkpoints
*.pth

# MMDetection3D experiment directories
work_dirs/

# Build artifacts
build/
dist/
*.egg-info/

# Python
__pycache__/
*.pyc
*.pyo

# Compiled files
*.pyd
*.dll
*.so

# TensorBoard
events.out.tfevents.*

# IDE
.vscode/
.idea/

# Windows
Thumbs.db

# macOS
.DS_Store
```

---

# 68. Scientific Project Context

This experiment forms the sensor-fusion component of a broader comparative 3D object-detection project.

The general comparison investigates:

```text
Camera-based 3D detection
          vs
LiDAR-based 3D detection
          vs
Camera + LiDAR fusion
```

The MVX-Net experiment represents:

```text
Camera + LiDAR Fusion
```

and uses the same KITTI object classes:

```text
Car
Pedestrian
Cyclist
```

for comparison with the other perception approaches.

---

# 69. Key Experiment Parameters

| Parameter | Value |
|---|---|
| Framework | MMDetection3D |
| MMDetection3D | 0.17.1 |
| Model | DynamicMVXFasterRCNN |
| Fusion | PointFusion |
| Dataset | KITTI |
| Sensors | Camera + LiDAR |
| Classes | Pedestrian, Cyclist, Car |
| Epochs | 80 |
| Batch size/GPU | 1 |
| Workers/GPU | 0 |
| Optimizer | AdamW |
| Initial LR | 0.003 |
| LR Scheduler | CosineAnnealing |
| Weight Decay | 0.01 |
| Gradient clipping | 35 |
| Image scale | 1280 × 384 |
| Voxel size | `[0.1, 0.1, 0.1]` |
| Point-cloud range | `[0, -40, -3, 70.4, 40, 1]` |
| Image backbone | ResNet-50 |
| Image neck | FPN |
| Voxel encoder | DynamicVFE |
| Fusion module | PointFusion |
| LiDAR encoder | SparseEncoder |
| LiDAR backbone | SECOND |
| LiDAR neck | SECONDFPN |
| Detection head | Anchor3DHead |
| Final checkpoint | `epoch_80.pth` |
| Prediction output | `mvxnet_epoch80_results.pkl` |
| Internal score threshold | 0.10 |
| Visualization threshold | 0.30 |
| Validation samples | 3769 |

---

# 70. Notes on Reproducibility

Exact numerical results can vary slightly due to:

- random initialization
- CUDA implementation details
- GPU architecture
- package versions
- random data augmentation
- non-deterministic CUDA operations

The logged experiment used:

```text
Random seed: 0
deterministic: False
```

Therefore bit-for-bit identical reproduction is not guaranteed.

For meaningful comparison, users should maintain the same:

```text
dataset split
configuration
class order
IoU thresholds
training schedule
evaluation protocol
software versions
```

---

# 71. Important Terminology

## Epoch

One complete training pass over the configured training dataset.

This experiment uses:

```text
80 epochs
```

`R40` in KITTI evaluation does **not** mean 40 epochs.

R40 refers to recall sampling used in KITTI Average Precision computation.

---

## Iteration

One optimizer update using one batch of data.

An epoch contains many iterations.

---

## Checkpoint

A saved training state such as:

```text
epoch_80.pth
```

---

## Inference

Using a trained model to make predictions without updating its weights.

---

## Validation

Evaluating the model on data excluded from training updates.

---

## AP

Average Precision.

AP summarizes the precision-recall relationship across confidence thresholds.

---

## IoU

Intersection over Union.

IoU measures the overlap between predicted and ground-truth bounding boxes.

---

## BEV

Bird's-Eye View.

A top-down representation of the scene.

---

## NMS

Non-Maximum Suppression.

Removes redundant overlapping predictions.

---

## Sensor Fusion

Combining information from multiple sensors.

In this project:

```text
Camera + LiDAR
```

---

# 72. Limitations

The experiment has several practical limitations:

- training was performed on a GTX 1650
- batch size was restricted by available GPU memory
- KITTI contains comparatively fewer Pedestrian and Cyclist instances than Car instances
- detection performance varies significantly across classes
- camera-LiDAR calibration must be accurate for effective PointFusion
- qualitative visualization depends on the selected confidence threshold
- the model and software stack are based on an older MMDetection3D release required for compatibility with the experiment environment

---

# 73. Future Work

Potential future improvements include:

- comparison against newer fusion architectures
- improved Pedestrian and Cyclist detection
- class-balanced training
- larger GPU training
- stronger image backbones
- additional augmentation
- adverse-weather evaluation
- night-time evaluation
- long-distance detection analysis
- ROS2 integration
- real-time perception deployment
- evaluation on additional datasets

---

# 74. Citation / Attribution

This project uses the MMDetection3D framework and an MVX-Net-style camera-LiDAR fusion architecture.

The repository represents an experimental implementation and evaluation performed for an academic scientific project.

The original MMDetection3D and model authors should be cited when this repository is used in academic work.

---

# 75. Summary

The complete experiment can be summarized as:

```text
KITTI Camera Images
        +
KITTI LiDAR Point Clouds
        │
        ▼
Dynamic MVX-Net
        │
        ├── ResNet-50
        ├── FPN
        ├── DynamicVFE
        ├── PointFusion
        ├── SparseEncoder
        ├── SECOND
        ├── SECONDFPN
        └── Anchor3DHead
        │
        ▼
80 Epoch Training
        │
        ▼
epoch_80.pth
        │
        ▼
MMDetection3D test.py
        │
        ▼
mvxnet_epoch80_results.pkl
        │
        ├── KITTI numerical evaluation
        │
        └── Custom visualization
                 │
                 ▼
         3D detection results
```

This repository provides the configuration, environment, evaluation outputs, and visualization tools required to understand and reproduce the experiment as closely as possible.
