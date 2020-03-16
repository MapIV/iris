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
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals = nullptr,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals = nullptr);

  void setPrePosition(const Eigen::Matrix4f& camera_pos_, const Eigen::Matrix4f& old_pos_, const Eigen::Matrix4f& older_pos_)
  {
    model_constraint = true;
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

  void setGain(double _scale, double _pitch, double _model, double _altitude)
  {
    scale_gain = _scale;
    pitch_gain = _pitch;
    model_gain = _model;
    altitude_gain = _altitude;
  }

private:
  bool model_constraint = false;
  double scale_gain = 1.0;
  double pitch_gain = 1.0;
  double model_gain = 1.0;
  double altitude_gain = 1.0;

  Eigen::Matrix4f camera_pos, old_pos, older_pos;

  void setVertexSim3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);
  void setVertexSE3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);

  void setEdge6DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals);

  void setEdge7DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      const pcl::Correspondences& correspondances,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
      const pcl::PointCloud<pcl::Normal>::Ptr& source_normals);
};
}  // namespace vllm