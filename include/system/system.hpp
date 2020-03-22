#pragma once
#include "core/bridge.hpp"
#include "core/config.hpp"
#include "core/types.hpp"
#include "core/util.hpp"
#include "map/map.hpp"
#include "map/parameter.hpp"
#include "optimize/optimizer.hpp"
#include "system/publisher.hpp"
#include <atomic>
#include <memory>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
class System
{
public:
  // ===== for Main ====
  System(Config& config, const std::shared_ptr<map::Map>& map);
  int execute();

public:
  // ==== for GUI ====
  cv::Mat getFrame() const { return bridge.getFrame(); }

  const std::shared_ptr<map::Map> getMap() const
  {
    return map;
  }

  void requestReset()
  {
    reset_requested.store(true);
  }

  bool popPublication(Publication& d)
  {
    return publisher.pop(d);
  }

  // void updateParameter()
  // {
  //   std::lock_guard<std::mutex> lock(parameter_mutex);
  //   parameter = thread_safe_parameter;
  // }

  // optimize::Gain getParameter() const
  // {
  //   std::lock_guard<std::mutex> lock(parameter_mutex);
  //   return parameter;
  // }
  // void setParameter(const optimize::Parameter& parameter_)
  // {
  //   std::lock_guard<std::mutex> lock(parameter_mutex);
  //   thread_safe_parameter = parameter_;
  // }

  // unsigned int getRecollection() const { return recollection; }
  // void setRecollection(unsigned int recollection_) { recollection = recollection_; }

private:
  bool optimize(int iteration);

  // ==== private member ====
  // optimize::Gain optimize_gain, thread_safe_optimize_gain;
  // mutable std::mutex optimize_gain_mutex;


  unsigned int recollection = 50;

  std::atomic<bool> reset_requested = false;

  const Config config;
  std::shared_ptr<map::Map> map;

  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f old_vllm_camera = Eigen::Matrix4f::Identity();    // t-1
  Eigen::Matrix4f older_vllm_camera = Eigen::Matrix4f::Identity();  // t-2

  // TODO: integrate
  std::vector<Eigen::Vector3f> vllm_trajectory;
  std::vector<Eigen::Vector3f> offset_trajectory;
  std::list<Eigen::Matrix4f> camera_history;

  pcl::CorrespondencesPtr correspondences;

  optimize::Config optimize_config;
  optimize::Optimizer optimizer;
  crrspEstimator estimator;

  BridgeOpenVSLAM bridge;
  double accuracy = 0.5;

  bool aligning_mode = false;
  map::Info localmap_info;

  Publisher publisher;

  // for relozalization
  bool relocalizing = false;
  const int history = 5;
  Eigen::Matrix4f camera_velocity;
};

}  // namespace vllm