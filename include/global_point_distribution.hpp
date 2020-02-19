// Global Point Distribution
#pragma once
#include "local_point_distribution.hpp"
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>

namespace vllm
{
class GPD
{
public:
  GPD(const size_t N, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float gain);
  LPD getLPD(const pcl::PointXYZ& point);

  std::vector<std::vector<std::vector<LPD>>> data;
  Eigen::Vector3f top;
  Eigen::Vector3f bottom;
  Eigen::Vector3f segment;
  const size_t N;

private:
  Eigen::Vector3i getIndex(const pcl::PointXYZ& point);

  Eigen::Vector3f getResolution(pcl::PointXYZ min, pcl::PointXYZ max, size_t N);
};
}  // namespace vllm