# VLLM
## Outline
* Visual Localization in 3D LiDAR Map
* It tracks the pose of monocular camera with respect to a given 3D LiDAR map

## Dependency
### can install with apt
* OpenCV
* PCL
* Eigen

### need to build from source
* [g2o](https://github.com/RainerKuemmerle/g2o)

## ThirdParty
* [Open-V-SLAM](https://github.com/xdspacelab/openvslam)
> submodule

## How to Build
* `-DBUILD_WITH_MARCH_NATIVE=ON`
> If your g2o is installed with `march_native=ON`

## References
* [Monocular Camera Localization in 3D LiDAR Maps](http://www.lifelong-navigation.eu/files/caselitz16iros.pdf)