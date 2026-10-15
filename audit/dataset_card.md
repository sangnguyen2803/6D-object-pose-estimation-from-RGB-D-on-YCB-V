---
license: mit
task_categories:
- object-detection
- image-segmentation
- robotics
dataset_info:
  features:
  - name: scene_id
    dtype: string
  - name: image_id
    dtype: string
  - name: obj_id
    dtype: int64
  - name: pose
    sequence:
      sequence: float64
  - name: camera_intrinsics
    sequence:
      sequence: float64
  - name: depth_scale
    dtype: float64
  - name: bbox
    sequence: int64
  - name: visibility
    dtype: float64
  - name: split
    dtype: string
  - name: rgb
    dtype: image
  - name: depth
    dtype: image
  - name: mask
    dtype: image
  - name: mask_visib
    dtype: image
  splits:
  - name: test
    num_bytes: 12240185177.56
    num_examples: 12247
  - name: train
    num_bytes: 8947085481.56
    num_examples: 10222
  download_size: 7105758283
  dataset_size: 21187270659.12
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: train
    path: data/train-*
---

# IndustryShapes

[**Project Page**](https://pose-lab.github.io/IndustryShapes) | [**Paper**](https://arxiv.org/abs/2602.05555)

IndustryShapes is a new benchmark dataset tailored for 6D object pose estimation in industrial settings. Targeting the challenges of textureless objects, reflective surfaces, and complex assembly tools, this dataset provides high-quality RGB-D data with precise annotations to advance the state of the art in robotic manipulation.

### Dataset Features
Unlike traditional datasets focused on household products, IndustryShapes introduces five new industry-relevant object types with challenging properties. The dataset features:
- **Realistic Settings:** Objects captured in authentic industrial assembly environments.
- **Diverse Complexity:** Scenes ranging from simple to challenging, including single and multiple objects, as well as multiple instances of the same object.
- **Unique Modalities:** It is the first dataset to offer RGB-D static onboarding sequences to support model-free and sequence-based approaches.
- **Comprehensive Annotations:** Includes high-quality annotated poses, bounding boxes, and segmentation masks.

### Dataset Organization
The dataset is organized into two parts:
- **Classic Set:** The Classic Set supports instance-level pose estimation with 21 scenes (13 train, 8 test). Includes images from real industrial scenes with varying complexity, Lab captured and Synthetically generated data.
- **Extended Set:** Inlucdes three challenging office scenes with unconstrained lighting, distractors, occlusions and diverse viewpoints featuring all objects. It also includes 10 **RGB-D** static onboarding sequences (2 per object).

### Tasks
- **6D Object Pose Estimation** (Instance-level and Novel Object)
- **Object Detection**
- **Image Segmentation**
- **Robotic Manipulation**