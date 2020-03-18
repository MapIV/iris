#pragma once
#include "bridge.hpp"
#include "config.hpp"
#include "map.hpp"
#include "parameter.hpp"
#include "types.hpp"
#include "util.hpp"
#include <atomic>
#include <memory>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
class System
{
public:
  // ===== for Main ====
  System(Config& config, const map::Map& map);
  int execute();
  bool optimize(int iteration);

public:
  // ==== for GUI ====
  cv::Mat getFrame() const { return bridge.getFrame(); }

  const pcXYZ::Ptr getTargetCloud() const
  {
    return map.getTargetCloud();
  }
  const pcNormal::Ptr getTargetNormals() const
  {
    return map.getTargetNormals();
  }

  void requestReset()
  {
    reset_requested.store(true);
  }

  bool popDatabase(Database& d)
  {
    return publisher.pop(d);
  }

  void updateParameter()
  {
    std::lock_guard<std::mutex> lock(parameter_mutex);
    parameter = thread_safe_parameter;
  }

  Parameter getParameter() const
  {
    std::lock_guard<std::mutex> lock(parameter_mutex);
    return parameter;
  }
  void setParameter(const Parameter& parameter_)
  {
    std::lock_guard<std::mutex> lock(parameter_mutex);
    thread_safe_parameter = parameter_;
  }

  // unsigned int getRecollection() const { return recollection; }
  // void setRecollection(unsigned int recollection_) { recollection = recollection_; }

private:
  // ==== private member ====
  float search_distance_min = 1;
  float search_distance_max = 10;
  Parameter parameter, thread_safe_parameter;
  mutable std::mutex parameter_mutex;

  unsigned int recollection = 50;

  std::atomic<bool> reset_requested = false;

  const Config config;
  map::Map map;

  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f old_vllm_camera = Eigen::Matrix4f::Identity();    // t-1
  Eigen::Matrix4f older_vllm_camera = Eigen::Matrix4f::Identity();  // t-2

  pcl::registration::CorrespondenceRejectorDistance distance_rejector;
  pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> estimator;

  BridgeOpenVSLAM bridge;
  double accuracy = 0.5;

  bool aligning_mode = false;

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