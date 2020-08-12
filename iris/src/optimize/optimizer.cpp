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

#include "optimize/optimizer.hpp"
#include "core/util.hpp"
#include "optimize/aligner.hpp"
#include <iostream>

namespace iris
{
namespace optimize
{
Outcome Optimizer::optimize(
    const std::shared_ptr<map::Map>& map_ptr,
    const pcXYZIN::Ptr& vslam_data,
    const Eigen::Matrix4f& offset_camera,
    crrspEstimator& estimator,
    const Eigen::Matrix4f& T_initial_align,
    const std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& vllm_history)
{
  pcXYZ::Ptr tmp_cloud(new pcXYZ);
  pcNormal::Ptr tmp_normals(new pcNormal);
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);

  Eigen::Matrix4f T_align = T_initial_align;

  auto distance_generator = [=](int itr) -> float {
    const float max = config.distance_max;
    const float min = config.distance_min;
    const float N = static_cast<float>(config.iteration);

    if (N <= 1)
      return min;

    return min + (N - 1 - static_cast<float>(itr)) * (max - min) / (N - 1);
  };

  for (int itr = 0; itr < config.iteration; itr++) {
    std::cout << "itration= \033[32m" << itr << "\033[m";

    util::transformXYZINormal(vslam_data, tmp_cloud, tmp_normals, T_align);

    // TODO: We should enable the estimator handle the PointXYZINormal
    estimator.setInputSource(tmp_cloud);
    estimator.setSourceNormals(tmp_normals);
    estimator.determineCorrespondences(*correspondences);

    std::cout << " ,raw_correspondences= \033[32m" << correspondences->size() << "\033[m";

    // NOTE: distance_rejector doesn't seemd to work well.
    // Reject too far correspondences
    float distance = distance_generator(itr);
    distance_rejector.setInputCorrespondences(correspondences);
    distance_rejector.setMaximumDistance(distance * distance);
    distance_rejector.getCorrespondences(*correspondences);
    std::cout << " ,refined_correspondecnes= \033[32m" << correspondences->size() << " @ " << distance << " \033[m" << std::endl;

    if (correspondences->size() < 10) {
      std::cout << "\033[33mSuspend optimization iteration because it have not enough correspondences\033[m" << std::endl;
      break;
    }


    Eigen::Matrix4f vllm_camera = T_align * offset_camera;
    Eigen::Matrix4f last_camera = vllm_camera;

    // Align pointclouds
    optimize::Aligner aligner(config.gain.scale, config.gain.latitude, config.gain.altitude, config.gain.smooth);
    T_align = aligner.estimate7DoF(
        T_align, vslam_data, map_ptr->getTargetCloud(), correspondences,
        offset_camera, vllm_history, config.ref_scale, map_ptr->getTargetNormals());

    // Integrate
    vllm_camera = T_align * offset_camera;

    // Get Inovation
    float scale = util::getScale(vllm_camera);
    float update_transform = (last_camera - vllm_camera).topRightCorner(3, 1).norm();        // called "Euclid distance"
    float update_rotation = (last_camera - vllm_camera).topLeftCorner(3, 3).norm() / scale;  // called "chordal distance"
    std::cout << "update= \033[33m" << update_transform << " \033[m,\033[33m " << update_rotation << "\033[m" << std::endl;

    std::cout << "T_align\n\033[4;36m"
              << T_align << "\033[m" << std::endl;

    if (config.threshold_translation > update_transform
        && config.threshold_rotation > update_rotation)
      break;
  }

  Outcome outcome;
  outcome.correspondences = correspondences;
  outcome.T_align = T_align;
  return outcome;
}


}  // namespace optimize
}  // namespace iris