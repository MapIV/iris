#pragma once
#include "bridge.hpp"
#include "config.hpp"
#include "util.hpp"
#include <atomic>
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
  int execute();
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
    return view_vllm_camera;
  }
  Eigen::Matrix4f getOffsetCamera() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    return view_offset_camera;
  }
  std::pair<pcXYZ::Ptr, pcl::CorrespondencesPtr> getAlignedCloudAndCorrespondences() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    pcXYZ::Ptr cloud(new pcXYZ);
    pcl::CorrespondencesPtr cors(new pcl::Correspondences);

    pcl::copyPointCloud(*view_vllm_cloud, *cloud);
    *cors = *correspondences_for_viewer;
    return {cloud, cors};
  }

  pcNormal::Ptr getAlignedNormals() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    pcNormal::Ptr normals(new pcNormal);
    pcl::copyPointCloud(*view_vllm_normals, *normals);
    return normals;
  }
  std::vector<Eigen::Vector3f> getOffsetTrajectory() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<Eigen::Vector3f> trajectory = offset_trajectory;
    return trajectory;
  }
  std::vector<Eigen::Vector3f> getTrajectory() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<Eigen::Vector3f> trajectory = vllm_trajectory;
    return trajectory;
  }

  void requestReset()
  {
    reset_requested.store(true);
  }

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
  double altitude_restriction_gain = 0;

  std::atomic<bool> reset_requested = false;
  unsigned int recollection = 50;

  Config config;

  mutable std::mutex mtx;

  bool first_set = true;
  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
  std::vector<Eigen::Vector3f> offset_trajectory;
  std::vector<Eigen::Vector3f> vllm_trajectory;

  Eigen::Matrix4f vllm_camera = Eigen::Matrix4f::Identity();        // t
  Eigen::Matrix4f old_vllm_camera = Eigen::Matrix4f::Identity();    // t-1
  Eigen::Matrix4f older_vllm_camera = Eigen::Matrix4f::Identity();  // t-2
  Eigen::Matrix4f offset_camera = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f view_vllm_camera = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f view_offset_camera = Eigen::Matrix4f::Identity();

  pcl::registration::CorrespondenceRejectorDistance distance_rejector;
  pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> estimator;

  BridgeOpenVSLAM bridge;
  double accuracy = 0.5;

  bool aligning_mode = false;

  Eigen::Matrix4f lost_point = Eigen::Matrix4f::Zero();

  pcl::PointCloud<pcl::PointXYZ>::Ptr view_vllm_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr offset_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;

  pcl::PointCloud<pcl::Normal>::Ptr target_normals;
  pcl::PointCloud<pcl::Normal>::Ptr source_normals;
  pcl::PointCloud<pcl::Normal>::Ptr offset_normals;
  pcl::PointCloud<pcl::Normal>::Ptr view_vllm_normals;

  bool relocalizing = false;
  const int history = 5;
  std::list<Eigen::Matrix4f> camera_history;
  Eigen::Matrix4f camera_velocity;

  pcl::CorrespondencesPtr correspondences;
  pcl::CorrespondencesPtr correspondences_for_viewer;
};

}  // namespace vllm