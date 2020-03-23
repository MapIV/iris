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

enum State {
  Inittializing = 0,
  Tracking = 1,
  Lost = 2,
  Relocalizing = 3,
};

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

  void updateOptimizeGain()
  {
    std::lock_guard<std::mutex> lock(optimize_gain_mutex);
    optimize_config.gain = thread_safe_optimize_gain;
  }
  optimize::Gain getOptimizeGain() const
  {
    std::lock_guard<std::mutex> lock(optimize_gain_mutex);
    return thread_safe_optimize_gain;
  }
  void setOptimizeGain(const optimize::Gain& gain_)
  {
    std::lock_guard<std::mutex> lock(optimize_gain_mutex);
    thread_safe_optimize_gain = gain_;
  }

  // unsigned int getRecollection() const { return recollection; }
  // void setRecollection(unsigned int recollection_) { recollection = recollection_; }

private:
  bool optimize(int iteration);

  // ==== private member ====
  optimize::Gain thread_safe_optimize_gain;
  optimize::Config optimize_config;
  optimize::Optimizer optimizer;
  mutable std::mutex optimize_gain_mutex;

  State state;


  unsigned int recollection = 50;

  std::atomic<bool> reset_requested = false;

  const Config config;
  std::shared_ptr<map::Map> map;

  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();

  std::vector<Eigen::Vector3f> vllm_trajectory;
  std::vector<Eigen::Vector3f> offset_trajectory;

  pcl::CorrespondencesPtr correspondences;

  crrspEstimator estimator;

  BridgeOpenVSLAM bridge;
  double accuracy = 0.5;

  map::Info localmap_info;

  Publisher publisher;

  // for relozalization
  Eigen::Matrix4f vllm_velocity;
  const int history = 5;
  std::list<Eigen::Matrix4f> vllm_history;
};

}  // namespace vllm