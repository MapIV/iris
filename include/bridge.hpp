#pragma once

#include "openvslam/config.h"
#include "openvslam/system.h"
#include <opencv2/videoio.hpp>

class BridgeOpenVSLAM
{
public:
  BridgeOpenVSLAM() = default;
  ~BridgeOpenVSLAM()
  {
    // wait until the loop BA is finished
    while (SLAM_ptr->loop_BA_is_running()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // shutdown the SLAM process
    SLAM_ptr->shutdown();
  }

  void setup(int argc, char* argv[]);
  bool execute();

  const std::shared_ptr<openvslam::publish::map_publisher> get_map_publisher()
  {
    return SLAM_ptr->get_map_publisher();
  }
  const std::shared_ptr<openvslam::publish::frame_publisher> get_frame_publisher()
  {
    return SLAM_ptr->get_frame_publisher();
  }

private:
  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;

  cv::VideoCapture video;

  bool is_not_end = true;
};