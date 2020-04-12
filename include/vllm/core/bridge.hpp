#pragma once
#include "openvslam/config.h"
#include "openvslam/system.h"
#include "vllm/core/config.hpp"
#include <opencv2/videoio.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
class BridgeOpenVSLAM
{
public:
  BridgeOpenVSLAM() {}
  ~BridgeOpenVSLAM();

  void setup(const Config& config);
  void execute(const cv::Mat& image);
  void requestReset();

  void getLandmarks(
      pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const;

  void getLandmarksAndNormals(
      pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
      pcl::PointCloud<pcl::Normal>::Ptr& normal,
      std::vector<float>& weights,
      unsigned int recollection,
      double accuracy) const;

  int getState() const;
  cv::Mat getFrame() const;
  Eigen::Matrix4d getCameraPose() const;
  unsigned int getPeriodFromInitialId();

private:
  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;
};
}  // namespace vllm