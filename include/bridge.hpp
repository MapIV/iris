#pragma once
#include "openvslam/config.h"
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

  void setup(int argc, char* argv[], const std::string& video_file_path, int frame_skip);
  bool execute();

  void getLandmarks(
      pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const;
  void getLandmarksAndNormals(
      pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
      pcl::PointCloud<pcl::Normal>::Ptr& normal) const;

  cv::Mat getFrame() const;

  Eigen::Matrix4d getCameraPose() const;

  int getState() const;


private:
  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;
  cv::VideoCapture video;

  int frame_skip = 1;
  bool is_not_end = true;
};