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
#include <g2o/core/sparse_optimizer.h>
#include <list>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_types.h>

namespace iris
{
namespace optimize
{
class Aligner
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Aligner(float scale_gain, float latitude_gain, float altitude_gain, float smooth_gain)
      : scale_gain(scale_gain),
        latitude_gain(latitude_gain),
        altitude_gain(altitude_gain),
        smooth_gain(smooth_gain) {}

  Aligner() : Aligner(0, 0, 0, 0) {}

  ~Aligner() {}

  Eigen::Matrix4f estimate7DoF(
      Eigen::Matrix4f& T,
      const pcXYZIN::Ptr& source_clouds,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::CorrespondencesPtr& correspondances,
      const Eigen::Matrix4f& offset_camera,
      const std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& history,
      const double ref_scale,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals = nullptr);

private:
  float scale_gain = 0;
  float latitude_gain = 0;
  float altitude_gain = 0;
  float smooth_gain = 0;

  void setVertexSim3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);
  void setVertexSE3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T);

  void setEdgeRestriction(
      g2o::SparseOptimizer& optimizer,
      const Eigen::Matrix4f& offset_camera,
      const Eigen::Matrix4f& T,
      double ref_scale);

  void setEdge7DoFGICP(
      g2o::SparseOptimizer& optimizer,
      const pcXYZIN::Ptr& source_clouds,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::CorrespondencesPtr& correspondances,
      const Eigen::Vector3f& camera,
      const pcl::PointCloud<pcl::Normal>::Ptr& target_normals);
};
}  // namespace optimize
}  // namespace iris