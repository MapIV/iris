#pragma once
#include "bridge.hpp"
#include "config.hpp"
#include "util.hpp"
#include <memory>
#include <mutex>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcNormal = pcl::PointCloud<pcl::Normal>;

class System
{
public:
  // ===== for Main ====
  System(int argc, char* argv[]);
  int update();
  bool optimize(int iteration);

public:
  // ==== for GUI ====
  cv::Mat getFrame() const { return bridge.getFrame(); }

  const pcXYZ::Ptr& getTargetCloud() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    return target_cloud;
  }
  const pcNormal::Ptr& getTargetNormals() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    return target_normals;
  }
  Eigen::Matrix4f getCamera() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    return vllm_camera;
  }
  Eigen::Matrix4f getRawCamera() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    return raw_camera;
  }
  pcXYZ::Ptr getAlignedCloud() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    pcXYZ::Ptr cloud(new pcXYZ);
    pcl::copyPointCloud(*aligned_cloud, *cloud);
    return cloud;
  }
  pcNormal::Ptr getAlignedNormals() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    pcNormal::Ptr normals(new pcNormal);
    pcl::copyPointCloud(*aligned_normals, *normals);
    return normals;
  }
  std::vector<Eigen::Vector3f> getRawTrajectory() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<Eigen::Vector3f> trajectory = raw_trajectory;
    return trajectory;
  }
  std::vector<Eigen::Vector3f> getTrajectory() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<Eigen::Vector3f> trajectory = vllm_trajectory;
    return trajectory;
  }

  // pcl::Correspondences getCorrespondences() const { return correspondences; }
  // void requestReset() { reset_requested = true; }
  // unsigned int getRecollection() const { return recollection; }
  // void setRecollection(unsigned int recollection_) { recollection = recollection_; }
  // Eigen::Vector3d getGain() const { return {scale_restriction_gain, pitch_restriction_gain, model_restriction_gain}; }
  // void setGain(const Eigen::Vector3d& gain)
  // {
  //   scale_restriction_gain = gain(0);
  //   pitch_restriction_gain = gain(1);
  //   model_restriction_gain = gain(2);
  // }

private:
  // ==== private member ====
  double search_distance_min = 0;
  double search_distance_max = 0;

  double scale_restriction_gain = 0;
  double pitch_restriction_gain = 0;
  double model_restriction_gain = 0;

  bool reset_requested = false;
  unsigned int recollection = 50;

  Eigen::Matrix4f last_vllm_camera = Eigen::Matrix4f::Identity();

  Config config;

  mutable std::mutex mtx;

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
  double accuracy = 0.5;

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr aligned_normals;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr source_normals;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr target_normals;

  pcl::CorrespondencesPtr correspondences;

  Eigen::Vector3f pre_camera = Eigen::Vector3f::Zero();
  Eigen::Vector3f pre_pre_camera = Eigen::Vector3f::Zero();
};

}  // namespace vllm