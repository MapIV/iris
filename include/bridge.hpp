#pragma once

#include "openvslam/config.h"
#include "openvslam/system.h"

class BridgeOpenVSLAM
{
public:
  BridgeOpenVSLAM() = default;

  void start(int argc, char* argv[]);

  const std::shared_ptr<openvslam::publish::map_publisher> get_map_publisher()
  {
    if (SLAM_ptr)
      return SLAM_ptr->get_map_publisher();
    else
      return nullptr;
  }
  const std::shared_ptr<openvslam::publish::frame_publisher> get_frame_publisher()
  {
    if (SLAM_ptr)
      return SLAM_ptr->get_frame_publisher();
    else
      return nullptr;
  }

private:
  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;

  void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
      const std::string& vocab_file_path, const std::string& video_file_path,
      const unsigned int frame_skip, const bool no_sleep, const bool auto_term);
};