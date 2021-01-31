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

#include "core/util.hpp"
#include "pcl_/normal_estimator.hpp"
#include <chrono>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <random>
#include <thread>

namespace iris
{

namespace util
{

namespace
{
float getScale_(const Eigen::MatrixXf& R)
{
  return static_cast<float>(std::sqrt((R.transpose() * R).trace() / 3.0));
}

// return the closest rotatin matrix
Eigen::Matrix3f normalize_(const Eigen::Matrix3f& R)
{
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f Vt = svd.matrixV().transpose();
  if (R.determinant() < 0) {
    return -U * Vt;
  }

  return U * Vt;
}
}  // namespace

Eigen::Matrix4f make3DPoseFrom2DPose(float x, float y, float nx, float ny)
{
  Eigen::Matrix4f T;
  T.setIdentity();

  T(0, 3) = x;
  T(1, 3) = y;
  float theta = std::atan2(ny, nx);
  Eigen::Matrix3f R;
  R << 0, 0, 1,
      -1, 0, 0,
      0, -1, 0;
  T.topLeftCorner(3, 3) = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()).toRotationMatrix() * R;
  return T;
}

Eigen::Matrix4f applyScaling(const Eigen::Matrix4f& T, float scale)
{
  Eigen::Matrix3f R = normalizeRotation(T);
  Eigen::Matrix4f scaled;
  scaled.setIdentity();
  scaled.rightCols(1) = T.rightCols(1);
  scaled.topLeftCorner(3, 3) = scale * R;
  return scaled;
}

// get scale factor from rotation matrix
float getScale(const Eigen::MatrixXf& A)
{
  if (A.cols() == 3)
    return getScale_(A);
  else if (A.cols() == 4)
    return getScale_(A.topLeftCorner(3, 3));
  return -1;
}

Eigen::Matrix3f normalizeRotation(const Eigen::MatrixXf& A)
{
  if (A.cols() != 3 && A.cols() != 4) {
    exit(1);
  }

  Eigen::Matrix3f sR = A.topLeftCorner(3, 3);
  float scale = getScale(sR);
  return normalize_(sR / scale);
}

Eigen::Matrix4f normalizePose(const Eigen::Matrix4f& sT)
{
  Eigen::Matrix4f T = sT;
  T.topLeftCorner(3, 3) = normalizeRotation(sT);
  return T;
}

void loadMap(
    const std::string& pcd_file,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    float grid_leaf, float radius)
{
  cloud->clear();
  normals->clear();

  // Load map pointcloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr all_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *all_cloud);

  // filtering
  if (grid_leaf > 0) {
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(all_cloud);
    filter.setLeafSize(grid_leaf, grid_leaf, grid_leaf);
    filter.filter(*cloud);
  } else {
    cloud = all_cloud;
  }

  if (grid_leaf > radius) {
    std::cerr << "normal_search_leaf" << radius << " must be larger than grid_leaf" << grid_leaf << std::endl;
    exit(EXIT_FAILURE);
  }

  // normal estimation
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl_::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setSearchSurface(all_cloud);
  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(*normals);
  std::cout << " normal " << normals->size() << ", points" << cloud->size() << ", surface " << all_cloud->size() << std::endl;

  return;
}


void transformNormals(const pcNormal& source, pcNormal& target, const Eigen::Matrix4f& T)
{
  Eigen::Matrix3f R = normalizeRotation(T);
  if (&source != &target) {
    target.clear();
    for (const pcl::Normal& n : source) {
      Eigen::Vector3f _n = R * n.getNormalVector3fMap();
      target.push_back({_n.x(), _n.y(), _n.z()});
    }
    return;
  }

  for (pcl::Normal& n : target) {
    Eigen::Vector3f _n = R * n.getNormalVector3fMap();
    n = pcl::Normal(_n.x(), _n.y(), _n.z());
  }
  return;
}

void transformXYZINormal(const pcXYZIN::Ptr& all, const pcXYZ::Ptr& points, const pcNormal::Ptr& normals, const Eigen::Matrix4f& T)
{
  points->clear();
  normals->clear();

  Eigen::Matrix3f sR = T.topLeftCorner(3, 3);
  Eigen::Matrix3f R = normalizeRotation(T);
  Eigen::Vector3f t = T.topRightCorner(3, 1);

  for (const xyzin& a : *all) {
    Eigen::Vector3f _p = sR * a.getVector3fMap() + t;
    Eigen::Vector3f _n = R * a.getNormalVector3fMap();

    points->push_back({_p.x(), _p.y(), _p.z()});
    normals->push_back({_n.x(), _n.y(), _n.z()});
  }
  return;
}

Eigen::Matrix3f randomRotation()
{
  return Eigen::Quaternionf::UnitRandom().toRotationMatrix();
}

void shufflePointCloud(pcXYZ::Ptr& cloud)
{
  std::mt19937 rand;
  for (size_t i = 0, size = cloud->size(); i < size; i++) {
    std::swap(cloud->points.at(i), cloud->points.at(rand() % size));
  }
}

}  // namespace util
}  // namespace iris