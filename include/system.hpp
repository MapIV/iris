#pragma once
#include "bridge.hpp"
#include "config.hpp"
#include "util.hpp"
#include <memory>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcNormal = pcl::PointCloud<pcl::Normal>;

class System
{
public:
  System(int argc, char* argv[]);

  int update();
  std::pair<float, float> optimize(int iteration);

  cv::Mat getFrame() const { return bridge.getFrame(); }

  const pcXYZ::Ptr& getAlignedCloud() const { return aligned_cloud; }
  const pcXYZ::Ptr& getTargetCloud() const { return target_cloud; }
  const pcl::CorrespondencesPtr& getCorrespondences() const { return correspondences; }

  const Eigen::Matrix4f& getCamera() const { return vllm_camera; }
  const Eigen::Matrix4f& getRawCamera() const { return raw_camera; }
  const std::vector<Eigen::Vector3f>& getTrajectory() const { return vllm_trajectory; }
  const std::vector<Eigen::Vector3f>& getRawTrajectory() const { return raw_trajectory; }

  const pcNormal::Ptr& getAlignedNormals() const { return aligned_normals; }
  const pcNormal::Ptr& getTargetNormals() const { return target_normals; }

  void requestReset() { reset_requested = true; }

  unsigned int getRecollection() const { return recollection; }
  void setRecollection(unsigned int recollection_) { recollection = recollection_; }
  Eigen::Vector2d getGain() const { return {scale_restriction_gain, pitch_restriction_gain}; }
  void setGain(const Eigen::Vector2d& gain)
  {
    scale_restriction_gain = gain(0);
    pitch_restriction_gain = gain(1);
  }

private:
  double scale_restriction_gain = 0;
  double pitch_restriction_gain = 0;
  bool reset_requested = false;
  unsigned int recollection = 50;

  Eigen::Matrix4f last_vllm_camera = Eigen::Matrix4f::Identity();

  int vslam_state;
  Config config;

  // setup for others
  bool first_set = true;
  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
  std::vector<Eigen::Vector3f> raw_trajectory;
  std::vector<Eigen::Vector3f> vllm_trajectory;
  Eigen::Matrix4f raw_camera = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f vllm_camera = Eigen::Matrix4f::Identity();

  pcl::registration::CorrespondenceRejectorDistance distance_rejector;
  pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> estimator;

  BridgeOpenVSLAM bridge;

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr aligned_normals;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr source_normals;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr target_normals;

  pcl::CorrespondencesPtr correspondences;
};

}  // namespace vllm