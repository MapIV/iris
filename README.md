**Modifications are underway.**


# *Iris*
* Visual localization in pre-build pointcloud maps.
* **OpenVSLAM** and **VINS-mono**  can be used.


## Video
[![](https://img.youtube.com/vi/a_BnifwBZC8/0.jpg)](https://www.youtube.com/watch?v=a_BnifwBZC8)


## Submodule 
* [OpenVSLAM forked by MapIV](https://github.com/MapIV/openvslam.git)
  * > [original repository (xdspacelab)](https://github.com/xdspacelab/openvslam)

## Dependency
If you are using ROS, you only need to install `g2o` and `DBoW2`.
* [ROS](http://wiki.ros.org/)
* [OpenCV](https://opencv.org/) >= 3.2
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) 
* [PCL](https://pointclouds.org/)
* [g2o](https://github.com/RainerKuemmerle/g2o)
* [DBow2](https://github.com/shinsumicco/DBoW2.git)
  * Please use the custom version released in [https://github.com/shinsumicco/DBoW2)](https://github.com/shinsumicco/DBoW2)

> see also: [openvslam](https://openvslam.readthedocs.io/en/master/installation.html#dependencies).

## How to Build
```bash
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive https://github.com/MapIV/iris.git
cd ..
catkin_make
```

## How to Run with Sample Data
### Download sample data
1. visual feature file: `orb_vocab.dbow` from [URL](https://www.dropbox.com/s/z8vodds9y6yxg0p/orb_vocab.dbow2?dl=0)
2. pointcloud map : `kitti_00.pcd` from [URL](https://www.dropbox.com/s/tzdqtsl1p7v1ylo/kitti_00.pcd?dl=0)
3. rosbag : `kitti_00_stereo.bag` from [URL](https://www.dropbox.com/s/kfouz9gkjefpvb5/kitti_00_stereo.bag?dl=0)

### Run with sample data
#### Monocular camera sample
```bash
roscd iris/../../../
# download sample data to here (orb_voceb.dbow, kitti_00.pcd, kitti_00_stereo.bag)
ls # > build devel install src orb_vocab.dbow kitti_00.pcd kitti_00_stereo.bag
roslaunch iris openvslam.launch
rosbag play kitti_00_stereo.bag # (on another terminal)
```
> If the estimated position is misaligned, it can be corrected using `2D Pose Estimate` in rviz.

#### Stereo camera sample
```bash
roscd iris/../../../
# download sample data to here (orb_voceb.dbow, kitti_00.pcd, kitti_00_stereo.bag)
ls # > build devel install src orb_vocab.dbow kitti_00.pcd kitti_00_stereo.bag
roslaunch iris stereo_openvslam.launch
rosbag play kitti_00_stereo.bag # (on another terminal)
```
> If the estimated position is misaligned, it can be corrected using `2D Pose Estimate` in rviz.



## How to Run with Your Data
### All you need to prepare
1. pointcloud map file (*.pcd)
1. rosbag (*.bag)
1. Config file for iris such as `config/sample_iris_config.yaml`
2. (only if you use OpenVSLAM) Config file for vSLAM such as `config/sample_openvslam_config.yaml` 
3. (only if you use VINS-mono) To use the VINS-mono, you need to write a launch file and a config file. (More detail in [https://github.com/HKUST-Aerial-Robotics/VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono))

### Run with OpenVSLAM
```bash
roslaunch iris openvslam.launch iris_config_path:=... 
rosbag play yours.bag # (on another terminal)
```
### Run with VINS-mono
```bash
roslaunch iris vinsmono.launch iris_config_path:=... 
roslaunch vins_estimator *something*.launch # (on another terminal)
rosbag play yours.bag # (on another terminal)
```

## License
**Modifications are underway.**

~~Iris is provided under the BSD 3-Clause License.~~

The following files are derived from third-party libraries.
* `iris/src/optimize/types_gicp.hpp` : part of [g2o](https://github.com/RainerKuemmerle/g2o) (BSD)
* `iris/src/optimize/types_gicp.cpp` : part of [g2o](https://github.com/RainerKuemmerle/g2o) (BSD)
* `iris/src/pcl_/correspondence_estimator.hpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
* `iris/src/pcl_/correspondence_estimator.cpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
* `iris/src/pcl_/normal_estimator.hpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
* `iris/src/pcl_/normal_estimator.cpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
