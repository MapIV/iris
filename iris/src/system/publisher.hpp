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
#include "core/util.hpp"
#include "map/info.hpp"
#include <Eigen/Dense>
#include <mutex>
#include <vector>

namespace iris
{
struct Publication {
  Publication() : cloud(new pcXYZ), normals(new pcNormal), correspondences(new pcl::Correspondences) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix4f iris_camera;
  Eigen::Matrix4f T_align;
  Eigen::Matrix4f offset_camera;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> iris_trajectory;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> offset_trajectory;
  map::Info localmap_info;

  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;
  pcl::CorrespondencesPtr correspondences;
};

// OBSL: thread safe publisher
class Publisher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Publisher()
  {
    flags[0] = flags[1] = false;
    id = 0;
  }

  bool pop(Publication& p);
  void push(
      const Eigen::Matrix4f& T_align,
      const Eigen::Matrix4f& iris_camera,
      const Eigen::Matrix4f& offset_camera,
      const pcXYZIN::Ptr& raw_data,
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& iris_trajectory,
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& offset_trajectory,
      const pcl::CorrespondencesPtr& corre,
      const map::Info& localmap_info);

private:
  int id;
  bool flags[2];
  Publication publication[2];
  mutable std::mutex mtx;
};

}  // namespace iris