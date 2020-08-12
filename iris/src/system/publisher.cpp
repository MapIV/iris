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

#include "system/publisher.hpp"

namespace iris
{
bool Publisher::pop(Publication& p)
{
  std::lock_guard<std::mutex> lock(mtx);

  if (flags[(id + 1) % 2] == false) {
    return false;
  }

  p = publication[(id + 1) % 2];
  flags[(id + 1) % 2] = false;
  return true;
}

// NOTE: There are many redundant copies
void Publisher::push(
    const Eigen::Matrix4f& T_align,
    const Eigen::Matrix4f& iris_camera,
    const Eigen::Matrix4f& offset_camera,
    const pcXYZIN::Ptr& raw_data,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& iris_trajectory,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& offset_trajectory,
    const pcl::CorrespondencesPtr& corre,
    const map::Info& localmap_info)
{
  Publication& tmp = publication[id];

  tmp.T_align = T_align;
  tmp.iris_camera = util::normalizePose(iris_camera);
  tmp.offset_camera = util::normalizePose(offset_camera);
  tmp.iris_trajectory = iris_trajectory;
  tmp.offset_trajectory = offset_trajectory;
  tmp.localmap_info = localmap_info;
  *tmp.correspondences = *corre;

  util::transformXYZINormal(raw_data, tmp.cloud, tmp.normals, T_align);

  {
    std::lock_guard<std::mutex> lock(mtx);
    flags[id] = true;
    id = (id + 1) % 2;
  }
}
}  // namespace iris