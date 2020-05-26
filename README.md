# *VLLM*
## Outline
* Visual Localization in prior LiDAR Maps
* It tracks the pose of monocular camera with respect to a given 3D LiDAR map
* **OpenVSLAM** and **VINS-mono**  can be used depending on the sensor configuration.
* **TODO**

## ThirdParty
* [Open-V-SLAM](https://github.com/xdspacelab/openvslam)

## Dependency
#### All on which openvslam depends
* please check [openvslam document](https://openvslam.readthedocs.io/en/master/installation.html#dependencies).
 
#### OpenCV
* [OpenCV](https://opencv.org/)

#### PCL
* [PCL](https://pointclouds.org/)

#### Eigen
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
#### g2o
* [g2o](https://github.com/RainerKuemmerle/g2o)


## How to Build
```bash
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive https://gitlab.com/MapIV/vllm.git
cd ..
catkin_make
```

## How to Run
### with OpenVSLAM
* **TODO**

### with VINS-mono
* **TODO**

## References
* [Monocular Camera Localization in 3D LiDAR Maps](http://www.lifelong-navigation.eu/files/caselitz16iros.pdf)