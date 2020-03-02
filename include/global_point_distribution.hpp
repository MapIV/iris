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
  GPD() {}
  GPD(size_t _N, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float gain) { init(_N, cloud, gain); }

  void init(size_t _N, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float gain);

  LPD getLPD(const pcl::PointXYZ& point);

  const LPD& at(size_t i, size_t j, size_t k) const
  {
    return data[i][j][k];
  }

  size_t size() const { return N; }

private:
  std::vector<std::vector<std::vector<LPD>>> data;
  Eigen::Vector3f top;
  Eigen::Vector3f bottom;
  Eigen::Vector3f segment;
  size_t N;

  Eigen::Vector3i getIndex(const pcl::PointXYZ& point);

  Eigen::Vector3f getResolution(pcl::PointXYZ min, pcl::PointXYZ max, size_t N);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace vllm