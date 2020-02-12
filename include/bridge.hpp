#pragma once

#include "openvslam/config.h"
#include "openvslam/data/landmark.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include "openvslam/system.h"
#include <opencv2/videoio.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

  void getLandmarks(
      pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const
  {
    std::vector<openvslam::data::landmark*> landmarks;
    std::set<openvslam::data::landmark*> local_landmarks;
    SLAM_ptr->get_map_publisher()->get_landmarks(landmarks, local_landmarks);

    if (landmarks.empty()) return;

    for (const auto lm : landmarks) {
      if (!lm || lm->will_be_erased()) {
        continue;
      }
      if (local_landmarks.count(lm)) {
        continue;
      }
      const openvslam::Vec3_t pos = lm->get_pos_in_world();
      pcl::PointXYZ p(
          static_cast<float>(pos.x()),
          static_cast<float>(pos.y()),
          static_cast<float>(pos.z()));

      cloud->push_back(p);
    }

    for (const auto local_lm : local_landmarks) {
      if (local_lm->will_be_erased()) {
        continue;
      }
      const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
      pcl::PointXYZ p(
          static_cast<float>(pos.x()),
          static_cast<float>(pos.y()),
          static_cast<float>(pos.z()));
      local_cloud->push_back(p);
    }
    return;
  }


  Eigen::Matrix4d getCameraPose() const
  {
    return SLAM_ptr->get_map_publisher()->get_current_cam_pose();
  }

  const std::shared_ptr<openvslam::publish::map_publisher> get_map_publisher() const
  {
    return SLAM_ptr->get_map_publisher();
  }
  const std::shared_ptr<openvslam::publish::frame_publisher> get_frame_publisher() const
  {
    return SLAM_ptr->get_frame_publisher();
  }
  const std::shared_ptr<openvslam::system> getSystem()
  {
    return SLAM_ptr;
  }

private:
  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;

  cv::VideoCapture video;

  int frame_skip = 1;


  bool is_not_end = true;
};