#pragma once
#include <opencv2/videoio.hpp>
#include <openvslam/system.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
class BridgeOpenVSLAM
{
public:
  BridgeOpenVSLAM() {}
  ~BridgeOpenVSLAM();

  void setup(const std::string& config_path, const std::string& vocab_path);
  void execute(const cv::Mat& image);
  void requestReset();

  void getLandmarksAndNormals(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& vslam_data) const;
  void setCriteria(unsigned int recollection_, float accuracy_);
  std::pair<unsigned int, float> getCriteria() const;

  // return openvslam::tracker_state_t
  int getState() const;

  cv::Mat getFrame() const;

  Eigen::Matrix4f getCameraPose() const;

private:
  unsigned int recollection = 0;
  float accuracy = -1;

  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;
};
}  // namespace vllm
