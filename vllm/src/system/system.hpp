#pragma once
#include "core/config.hpp"
#include "core/types.hpp"
#include "core/util.hpp"
#include "map/map.hpp"
#include "map/parameter.hpp"
#include "optimize/optimizer.hpp"
#include "system/publisher.hpp"
#include <atomic>
#include <memory>

namespace vllm
{
enum VllmState {
  Inittializing = 0,
  Tracking = 1,
  Lost = 2,
  Relocalizing = 3,
};

class System
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // ===== for Main ====
  System(const Config& config_, const std::shared_ptr<map::Map>& map_);

  int execute(int vslam_state, const Eigen::Matrix4f& T_vslam, const pcXYZIN::Ptr& vslam_data);

public:
  Eigen::Matrix4f getT() const
  {
    return T_world;
  }

  void setImuPrediction(const Eigen::Matrix4f& T_world_)
  {
    T_imu = T_world_;
  }

  const std::shared_ptr<map::Map> getMap() const
  {
    return map;
  }

  void requestReset()
  {
    reset_requested.store(true);
  }

  bool popPublication(Publication& p)
  {
    return publisher.pop(p);
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

  unsigned int getRecollection() const
  {
    return recollection.load();
  }

  void setRecollection(unsigned int recollection_)
  {
    recollection.store(recollection_);
  }

  Eigen::Matrix4f getTWorld() const
  {
    return T_world;
  }

  // TODO:NOTE: This is not thread safe
  void specifyTWorld(const Eigen::Matrix4f& specified_T_world)
  {
    // std::cout << "last T_align\n"
    //           << T_align << std::endl;
    // std::cout << "last T_world\n"
    //           << T_world << std::endl;
    // std::cout << "last T_vslam\n"
    //           << last_T_vslam << std::endl;

    float scale = util::getScale(T_world);
    auto scaled_new_T_world = util::applyScaling(specified_T_world, scale);
    // std::cout << "scaled_new_T_world\n"
    //           << scaled_new_T_world << std::endl;
    T_align = scaled_new_T_world * last_T_vslam.inverse();

    // std::cout << "new T_align\n"
    //           << T_align << std::endl;
  }

  // TODO:NOTE: This is not thread safe.
  void specifyScale(const float scale)
  {
    T_align = util::applyScaling(T_align, scale);
  }

private:
  void updateOptimizeGain()
  {
    std::lock_guard<std::mutex> lock(optimize_gain_mutex);
    optimize_config.gain = thread_safe_optimize_gain;
  }

  // ==== private member ====
  optimize::Gain thread_safe_optimize_gain;
  optimize::Config optimize_config;
  optimize::Optimizer optimizer;
  mutable std::mutex optimize_gain_mutex;

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> vllm_trajectory;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> offset_trajectory;

  VllmState vllm_state;

  std::atomic<unsigned int> recollection;

  std::atomic<bool> reset_requested = false;

  Config config;
  std::shared_ptr<map::Map> map;

  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_world = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_imu = Eigen::Matrix4f::Zero();
  Eigen::Matrix4f last_T_vslam = Eigen::Matrix4f::Identity();

  pcl::CorrespondencesPtr correspondences;
  crrspEstimator estimator;

  map::Info localmap_info;
  Publisher publisher;


  // for relozalization
  Eigen::Matrix4f vllm_velocity;
  const int history = 30;
  std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> vllm_history;
};

}  // namespace vllm