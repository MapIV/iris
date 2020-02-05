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

### optinal
* Pangolin
* Socket Viewer

## ThirdParty
* [Open-V-SLAM](https://github.com/xdspacelab/openvslam)
> submodule


## How to Build
* `-DBUILD_WITH_MARCH_NATIVE=OFF`
> Installing g2o with `march_native=ON` is recommended.
    
``` bash
cmake \
    -DBUILD_WITH_MARCH_NATIVE=OFF \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBOW_FRAMEWORK=DBoW2 \
    -DBUILD_TESTS=OFF \
    ..
```
``` bash
./run_video_slam -v ./orb_vocab/orb_vocab.dbow2 -m ../../room-long.mp4  -c ./room/config.yaml --frame-skip 3 --no-sleep --map-db map.msg
```

## References
* [Monocular Camera Localization in 3D LiDAR Maps](http://www.lifelong-navigation.eu/files/caselitz16iros.pdf)

