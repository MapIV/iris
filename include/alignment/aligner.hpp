#pragma once
#include "alignment/parameter.hpp"
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

  void setPrePosition(const Eigen::Matrix4f& camera_pos_, const Eigen::Matrix4f& old_pos_, const Eigen::Matrix4f& older_pos_)
  {
    camera_pos = camera_pos_;
    old_pos = old_pos_;
    older_pos = older_pos_;
  }

  Eigen::Matrix4f estimate7DoF(
      Eigen::Matrix4f& T,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals = nullptr,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals = nullptr);

  void setParameter(const Parameter& param_)
  {
    param = param_;
  }

private:
  Parameter param;

  Eigen::Matrix4f camera_pos, old_pos, older_pos;

  void setVertexSim3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);
  void setVertexSE3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);

  void setEdge7DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals);
};
}  // namespace vllm