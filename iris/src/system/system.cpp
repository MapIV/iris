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

#include "system/system.hpp"
#include "optimize/aligner.hpp"
#include "optimize/averager.hpp"
#include "optimize/optimizer.hpp"
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>

namespace iris
{

System::System(const Config& config_, const std::shared_ptr<map::Map>& map_)
{
  config = config_;
  correspondences = pcl::CorrespondencesPtr(new pcl::Correspondences);
  map = map_;

  // System starts in initialization mode
  iris_state = IrisState::Inittializing;

  // Setup correspondence estimator
  estimator.setInputTarget(map->getTargetCloud());
  estimator.setTargetNormals(map->getTargetNormals());
  estimator.setKSearch(10);

  localmap_info = map->getLocalmapInfo();


  T_world = config.T_init;
  T_align.setIdentity();
  iris_velocity.setZero();

  // for optimizer module
  optimize::Gain gain;
  gain.scale = config.scale_gain;
  gain.smooth = config.smooth_gain;
  gain.altitude = config.altitude_gain;
  gain.latitude = config.latitude_gain;

  optimize_config.gain = gain;
  optimize_config.distance_min = config.distance_min;
  optimize_config.distance_max = config.distance_max;
  optimize_config.iteration = config.iteration;
  optimize_config.threshold_rotation = config.converge_rotation;
  optimize_config.threshold_translation = config.converge_translation;
  optimize_config.ref_scale = util::getScale(config.T_init);

  // During the constructor funtion, there is no way to access members from other threads
  thread_safe_optimize_gain = optimize_config.gain;

  for (int i = 0; i < history; i++)
    iris_history.push_front(T_world);
}

int System::execute(int vslam_state, const Eigen::Matrix4f& T_vslam, const pcXYZIN::Ptr& vslam_data)
{
  // ====================
  if (iris_state == IrisState::Inittializing) {
    // NOTE: "2" means openvslam::tracking_state_t::Tracking
    if (vslam_state == 2) iris_state = IrisState::Tracking;
    T_align = T_world;
  }

  // =======================
  if (iris_state == IrisState::Lost) {
    std::cerr << "\033[31mIrisState::Lost has not been implemented yet.\033[m" << std::endl;
    exit(1);
  }

  // ====================
  if (iris_state == IrisState::Relocalizing) {
    std::cerr << "\033[31mIrisState::Relocalizing has not been implemented yet.\033[m" << std::endl;
    exit(1);
  }

  // ====================
  if (iris_state == IrisState::Tracking) {
    // Optimization
    updateOptimizeGain();
    optimizer.setConfig(optimize_config);
    Eigen::Matrix4f T_initial_align = T_align;

    optimize::Outcome outcome = optimizer.optimize(
        map, vslam_data, T_vslam, estimator, T_initial_align, iris_history);
    correspondences = outcome.correspondences;
    T_align = outcome.T_align;
  }

  // update the pose in the world
  T_world = T_align * T_vslam;

  // Update local map
  map->informCurrentPose(T_world);
  map::Info new_localmap_info = map->getLocalmapInfo();

  // Reinitialize correspondencesEstimator
  if (localmap_info != new_localmap_info) {
    localmap_info = new_localmap_info;
    correspondences->clear();
    estimator.setInputTarget(map->getTargetCloud());
    estimator.setTargetNormals(map->getTargetNormals());
  }

  // Update history
  iris_history.pop_back();
  iris_history.push_front(T_world);

  // Pubush for the viewer
  iris_trajectory.push_back(T_world.topRightCorner(3, 1));
  offset_trajectory.push_back((config.T_init * T_vslam).topRightCorner(3, 1));
  publisher.push(
      T_align, T_world, config.T_init * T_vslam,
      vslam_data, iris_trajectory,
      offset_trajectory, correspondences, localmap_info);

  last_T_vslam = T_vslam;

  return iris_state;
}
}  // namespace iris