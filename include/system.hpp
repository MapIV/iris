#pragma once
#include "bridge.hpp"
#include "config.hpp"
#include "type.hpp"
#include "util.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
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

  void requestReset()
  {
    reset_requested.store(true);
  }

  bool popDatabase(Database& d)
  {
    return publisher.pop(d);
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
  unsigned int recollection = 50;

  std::atomic<bool> reset_requested = false;

  Config config;

  mutable std::mutex mtx;

  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f old_vllm_camera = Eigen::Matrix4f::Identity();    // t-1
  Eigen::Matrix4f older_vllm_camera = Eigen::Matrix4f::Identity();  // t-2

  pcl::registration::CorrespondenceRejectorDistance distance_rejector;
  pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> estimator;

  BridgeOpenVSLAM bridge;
  double accuracy = 0.5;

  bool aligning_mode = false;

  pcXYZ::Ptr source_cloud;
  pcXYZ ::Ptr target_cloud;
  pcNormal::Ptr source_normals;
  pcNormal ::Ptr target_normals;

  // database
  Database database;
  Publisher publisher;

  // for relozalization
  bool relocalizing = false;
  const int history = 5;
  std::list<Eigen::Matrix4f> camera_history;
  Eigen::Matrix4f camera_velocity;
};

}  // namespace vllm