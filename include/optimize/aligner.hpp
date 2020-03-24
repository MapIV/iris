#pragma once
#include <g2o/core/sparse_optimizer.h>
#include <list>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_types.h>

namespace vllm
{
namespace optimize
{
class Aligner
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Aligner(float scale_gain, float latitude_gain, float altitude_gain, float smooth_gain)
      : scale_gain(scale_gain),
        latitude_gain(latitude_gain),
        altitude_gain(altitude_gain),
        smooth_gain(smooth_gain) {}

  Aligner() : Aligner(0, 0, 0, 0) {}

  ~Aligner() {}

  Eigen::Matrix4f estimate7DoF(
      Eigen::Matrix4f& T,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::CorrespondencesPtr& correspondances,
      const Eigen::Matrix4f& offset_camera,
      const std::list<Eigen::Matrix4f>& history,
      const std::vector<float>& weights,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals = nullptr,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals = nullptr);

private:
  float scale_gain = 0;
  float latitude_gain = 0;
  float altitude_gain = 0;
  float smooth_gain = 0;

  void setVertexSim3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);
  void setVertexSE3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);

  void setEdgeRestriction(
      g2o::SparseOptimizer& optimizer,
      const Eigen::Matrix4f& offset_camera,
      const std::list<Eigen::Matrix4f>& history);

  void setEdge7DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::CorrespondencesPtr& correspondances,
      const std::vector<float>& weights,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals);
};
}  // namespace optimize
}  // namespace vllm