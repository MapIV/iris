#pragma once

#include "openvslam/config.h"
#include "openvslam/system.h"
#include <atomic>

class BridgeOpenVSLAM
{
  std::atomic<bool> slam_ready;

public:
  BridgeOpenVSLAM() : slam_ready(false) {}

  void start(int argc, char* argv[]);

  const std::shared_ptr<openvslam::publish::map_publisher> get_map_publisher()
  {
    while (slam_ready.load() == false) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return SLAM_ptr->get_map_publisher();
  }
  const std::shared_ptr<openvslam::publish::frame_publisher> get_frame_publisher()
  {
    while (slam_ready.load() == false) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return SLAM_ptr->get_frame_publisher();
  }

private:
  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;


  void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
      const std::string& vocab_file_path, const std::string& video_file_path,
      const unsigned int frame_skip, const bool no_sleep, const bool auto_term);
};