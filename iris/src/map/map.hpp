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
#include "core/types.hpp"
#include "core/util.hpp"
#include "map/info.hpp"
#include "map/parameter.hpp"
#include <atomic>
#include <fstream>
#include <mutex>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <sstream>
#include <unordered_map>

namespace iris
{
namespace map
{

struct HashForPair {
  template <typename T1, typename T2>
  size_t operator()(const std::pair<T1, T2>& p) const
  {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);

    // https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
    size_t seed = 0;
    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

class Map
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Map(const Parameter& parameter, const Eigen::Matrix4f& T_init);

  // If the map updates then return true.
  bool informCurrentPose(const Eigen::Matrix4f& T);

  // This informs viewer of whether local map updated or not
  Info getLocalmapInfo() const
  {
    return localmap_info;
  }

  const pcXYZ::Ptr getTargetCloud() const
  {
    return local_target_cloud;
  }

  const pcXYZ::Ptr getSparseCloud() const
  {
    return all_sparse_cloud;
  }

  const pcNormal::Ptr getTargetNormals() const
  {
    return local_target_normals;
  }

private:
  const std::string cache_file;
  const Parameter parameter;
  const std::string cache_cloud_file = "iris_cloud.pcd";
  const std::string cache_normals_file = "iris_normals.pcd";
  const std::string cache_sparse_file = "iris_sparse_cloud.pcd";

  // whole point cloud
  pcXYZ::Ptr all_target_cloud;
  pcNormal::Ptr all_target_normals;
  pcXYZ::Ptr all_sparse_cloud;

  // valid point cloud
  pcXYZ::Ptr local_target_cloud;
  pcNormal::Ptr local_target_normals;

  // divided point cloud
  std::unordered_map<std::pair<int, int>, pcXYZ, HashForPair> submap_cloud;
  std::unordered_map<std::pair<int, int>, pcNormal, HashForPair> submap_normals;

  // [x,y,theta]
  Eigen::Vector3f last_grid_center;
  Info localmap_info;


  bool isRecalculationNecessary() const;
  bool isUpdateNecessary(const Eigen::Matrix4f& T) const;
  void updateLocalmap(const Eigen::Matrix4f& T);

  // return [0,2*pi]
  float yawFromPose(const Eigen::Matrix4f& T) const;

  // return [0,pi]
  float subtractAngles(float a, float b) const
  {
    // a,b \in [0,2\pi]
    float d = std::fabs(a - b);
    if (d > 3.14159f)
      return 2.f * 3.14159f - d;
    return d;
  }
};
}  // namespace map
}  // namespace iris