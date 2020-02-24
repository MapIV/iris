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

  Eigen::Matrix4f estimate6DoF(
      Eigen::Matrix4f& T,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals = nullptr);  // for target

  Eigen::Matrix4f estimate7DoF(
      Eigen::Matrix4f& T,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals = nullptr);  // for target

private:
  void setVertexSim3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);
  void setVertexSE3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);

  void setEdge6DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals);  // for target

  void setEdge7DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals);  // for target
};
}  // namespace vllm