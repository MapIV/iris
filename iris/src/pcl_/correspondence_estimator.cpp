/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */
#ifndef PCL_REGISTRATION_IMPL_CORRESPONDENCE_ESTIMATION_BACK_PROJECTION_HPP_
#define PCL_REGISTRATION_IMPL_CORRESPONDENCE_ESTIMATION_BACK_PROJECTION_HPP_

#include "pcl_/correspondence_estimator.hpp"
#include <pcl/common/copy_point.h>

namespace iris
{
namespace pcl_
{
// Set epsilon > 1 when the normal vector is considered
Eigen::Matrix3f calcInversedCovariance(const Eigen::Vector3f& n, float epsilon = 1.0f)
{
  Eigen::Vector3f n0 = n.normalized();
  Eigen::Vector3f e = Eigen::Vector3f::UnitX();
  if (e.dot(n0) > 1 - 1e-6) e = Eigen::Vector3f::UnitY();

  Eigen::Vector3f n1 = (e - e.dot(n0) * n0).normalized();
  Eigen::Vector3f n2 = n0.cross(n1);

  Eigen::Matrix3f R;
  //      |n1_x, n1_y, n1_z|
  // R =  |n2_x, n2_y, n2_z|
  //      |n_x , n_y , n_z|
  R.row(0) = n1.transpose();
  R.row(1) = n2.transpose();
  R.row(2) = n0.transpose();

  // clang-format off
  Eigen::Matrix3f inv_cov;
  inv_cov <<   epsilon,      0,       0,
                     0,epsilon,       0,
                     0,      0,       1;
  // clang-format on

  return R.transpose() * inv_cov * R;
}


///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename NormalT, typename Scalar>
bool CorrespondenceEstimationBackProjection<PointSource, PointTarget, NormalT, Scalar>::initCompute()
{
  if (!source_normals_ || !target_normals_) {
    PCL_WARN("[pcl::registration::%s::initCompute] Datasets containing normals for source/target have not been given!\n", getClassName().c_str());
    return (false);
  }

  return (pcl::registration::CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>::initCompute());
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename NormalT, typename Scalar>
void CorrespondenceEstimationBackProjection<PointSource, PointTarget, NormalT, Scalar>::determineCorrespondences(
    pcl::Correspondences& correspondences, double max_distance)
{
  if (!initCompute())
    return;

  correspondences.resize(indices_->size());

  std::vector<int> nn_indices(k_);
  std::vector<float> nn_dists(k_);

  float min_dist = std::numeric_limits<float>::max();
  int min_index = 0;
  float min_output_dist = 0;

  pcl::Correspondence corr;
  unsigned int nr_valid_correspondences = 0;

  // Check if the template types are the same. If true, avoid a copy.
  // Both point types MUST be registered using the POINT_CLOUD_REGISTER_POINT_STRUCT macro!
  if (pcl::isSamePointType<PointSource, PointTarget>()) {

    PointTarget pt;
    // Iterate over the input set of source indices
    for (std::vector<int>::const_iterator idx_i = indices_->begin(); idx_i != indices_->end(); ++idx_i) {

      min_dist = std::numeric_limits<float>::max();
      Eigen::Vector3f input_point = input_->points[*idx_i].getVector3fMap();
      // Eigen::Vector3f input_normal = source_normals_->points[*idx_i].getNormalVector3fMap();

      Eigen::Vector3f offset_point = input_point;
      tree_->nearestKSearch(PointSource(offset_point.x(), offset_point.y(), offset_point.z()), k_, nn_indices, nn_dists);

      // Find the best correspondence
      for (size_t j = 0; j < nn_indices.size(); j++) {
        Eigen::Vector3f target_point = target_->points[nn_indices[j]].getVector3fMap();
        Eigen::Vector3f target_normal = target_normals_->points[nn_indices[j]].getNormalVector3fMap();
        Eigen::Matrix3f information_matrix = calcInversedCovariance(target_normal, 0.1f);

        Eigen::Vector3f e = target_point - input_point;
        float dist = e.dot(information_matrix * e);

        if (dist < min_dist) {
          min_dist = dist;
          min_index = nn_indices[j];
          min_output_dist = nn_dists[j];
        }
      }

      if (min_dist > max_distance)
        continue;

      corr.index_query = *idx_i;
      corr.index_match = min_index;
      corr.distance = min_output_dist;
      correspondences[nr_valid_correspondences++] = corr;
    }
  } else {
    PCL_WARN("called the NOT implemented function in CorrespondenceEstimationBackprojection::determinCorrespondence!\n", getClassName().c_str());
  }
  correspondences.resize(nr_valid_correspondences);
  deinitCompute();
}
template class CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal>;

}  // namespace pcl_
}  // namespace iris

#endif  // PCL_REGISTRATION_IMPL_CORRESPONDENCE_ESTIMATION_BACK_PROJECTION_HPP_
