// Copyright (c) 2020, Map IV, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the Map IV, Inc. nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

namespace iris
{
enum IrisState {
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

  Eigen::Matrix4f getTWorld() const
  {
    return T_world;
  }

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

    std::cout << "new T_align\n"
              << T_align << std::endl;
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

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> iris_trajectory;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> offset_trajectory;

  IrisState iris_state;


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
  Eigen::Matrix4f iris_velocity;
  const int history = 30;
  std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> iris_history;
};

}  // namespace iris