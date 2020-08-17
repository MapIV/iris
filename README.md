# *Iris*
## Outline
* visual localization in prior LiDAR maps
* It tracks the pose of monocular camera with respect to a given 3D LiDAR map
* **OpenVSLAM** and **VINS-mono**  can be used depending on the sensor configuration.

[![](https://img.youtube.com/vi/a_BnifwBZC8/0.jpg)](https://www.youtube.com/watch?v=a_BnifwBZC8)

## Submodule 
* [OpenVSLAM forked by MapIV](https://github.com/MapIV/openvslam.git)
> [original](https://github.com/xdspacelab/openvslam)

## Dependency
please check [openvslam document](https://openvslam.readthedocs.io/en/master/installation.html#dependencies).
 
* [OpenCV](https://opencv.org/)
* [PCL](https://pointclouds.org/)
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [g2o](https://github.com/RainerKuemmerle/g2o)

## How to Build
```bash
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive https://MapIV/iris.git
cd ..
catkin_make
```

## How to Run
### with OpenVSLAM
```bash
roslaunch iris openvslam.launch
```
### with VINS-mono
install VINS-mono
```bash
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone 
cd ..
catkin_make
```

run with VINS-mono
```bash
roslaunch iris vinsmono.launch
roslaunch vins_estimator *something*.launch
```

## Run with sample data
### with OpenVSLAM
```bash
roscd iris/../../../
# download sample data (hongo.pcd, orb_vocab.dbow2, honog-theta.bag)
roslaunch iris openvslam.launch
rosbag play hongo-theta.bag # (on another terminal)
```

### with VINS-mono
```bash
roscd iris/../../../
# download sample data (hongo.pcd, orb_vocab.dbow2, honog-imu.bag)
roslaunch iris vinsmono.launch
roslaunch vins_estimator realsense_color.launch # (on another terminal)
rosbag play hongo-imu.bag # (on another terminal)
```

## License
Iris is provided under the BSD 3-Clause License.

The following files are derived from third-party libraries.
* `iris/src/optimize/types_gicp.hpp` : part of [g2o](https://github.com/RainerKuemmerle/g2o) (BSD)
* `iris/src/optimize/types_gicp.cpp` : part of [g2o](https://github.com/RainerKuemmerle/g2o) (BSD)
* `iris/src/pcl_/correspondence_estimator.hpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
* `iris/src/pcl_/correspondence_estimator.cpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
* `iris/src/pcl_/normal_estimator.hpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)
* `iris/src/pcl_/normal_estimator.cpp` : part of [pcl](https://github.com/PointCloudLibrary/pcl) (BSD)

## References
* [Monocular Camera Localization in 3D LiDAR Maps](http://www.lifelong-navigation.eu/files/caselitz16iros.pdf)