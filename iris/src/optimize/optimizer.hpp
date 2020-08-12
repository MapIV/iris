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
#include "core/keypoints_with_normal.hpp"
#include "core/types.hpp"
#include "map/map.hpp"
#include <Eigen/Dense>

namespace iris
{
namespace optimize
{
struct Gain {
  // for Solver
  float scale = 0;
  float latitude = 0;
  float altitude = 0;
  float smooth = 0;
};

struct Config {
  // for Optimizer
  float threshold_translation = 0;
  float threshold_rotation = 0;
  float distance_max;
  float distance_min;
  int iteration;
  float ref_scale = 1;
  Gain gain;
};

struct Outcome {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::CorrespondencesPtr correspondences = nullptr;
  Eigen::Matrix4f T_align;
};

class Optimizer
{
public:
  void setConfig(const Config& config_) { config = config_; }

  Outcome optimize(
      const std::shared_ptr<map::Map>& map_ptr,
      const pcXYZIN::Ptr& vslam_data,
      const Eigen::Matrix4f& offset_camera,
      crrspEstimator& estimator,
      const Eigen::Matrix4f& T_initial_align,
      const std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& vllm_history);

private:
  Config config;
  crrspRejector distance_rejector;
};
}  // namespace optimize
}  // namespace iris