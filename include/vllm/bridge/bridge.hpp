#pragma once
#include "openvslam/system.h"
#include "vllm/core/config.hpp"
#include "vllm/core/types.hpp"
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

  void getLandmarksAndNormals(pcXYZ::Ptr& local_cloud, pcNormal::Ptr& normal, std::vector<float>& weights) const;
  void setCriteria(unsigned int recollection_, float accuracy_);
  std::pair<unsigned int, float> getCriteria() const;

  // return openvslam::tracker_state_t
  int getState() const;

  cv::Mat getFrame() const;

  Eigen::Matrix4f getCameraPose() const;

  // you can util it when vslam get recover
  unsigned int getPeriodFromInitialId() const;

private:
  unsigned int recollection = 0;
  float accuracy = -1;

  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;
};
}  // namespace vllm