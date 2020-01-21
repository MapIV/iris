#pragma once
#include <g2o/core/sparse_optimizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_types.h>

namespace vllm
{
class Aligner
{
public:
  Aligner() {}

  ~Aligner() {}

  void estimate(
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      Eigen::Matrix4f& T);

private:
  void setVertexSim3(g2o::SparseOptimizer& optimizer);

  void setEdgeGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances);
};
}  // namespace vllm