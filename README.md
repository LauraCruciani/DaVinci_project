# DaVinci_project
## Introduction
### Project Description

The project aims to reconstruct the surgical scene in 3D acquired from the Da Vinci system, segmenting and highlighting the common iliac artery.
<center>
  <img src="Result.gif" alt="Alt Result">
</center>
The repository is organized as follows:

#### 1. Calibration
In this folder, you can find code to acquire calibration images from the calibration videos and code to obtain calibration parameters from the frames. Additionally, there is a sample dataset included.

#### 2. Offline
In this folder, you can find code for 3D reconstruction and segmentation of images from a dataset.

#### 3. Online
In this folder, you can find code for real-time 3D reconstruction and segmentation of vessels using images from the Da Vinci system.

#### Main.py
##### Stereo Vision Point Cloud Generation

This project utilizes stereo vision techniques to extract frames from two synchronized videos, compute disparities, and generate a 3D point cloud representation of the scene. It also performs image rectification and re-projection to align the stereo images, facilitating accurate depth calculations and point cloud generation.

##### Features

- **Frame Extraction**: Extracts frames from left and right video inputs.
- **Image Rectification**: Aligns stereo images to correct for camera distortions.
- **Disparity Calculation**: Computes disparity maps using a high-resolution stereo model.
- **Point Cloud Generation**: Creates 3D point clouds from the computed disparity maps.
- **Point Cloud Registration**: Aligns the generated point clouds with a source point cloud for further analysis.
- **Image Re-projection**: Projects the registered point cloud back to 2D images.

