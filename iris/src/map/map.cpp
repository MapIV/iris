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

#include "map/map.hpp"
#include "core/util.hpp"
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <utility>

namespace iris
{

namespace map
{
Map::Map(const Parameter& parameter, const Eigen::Matrix4f& T_init)
    : cache_file("iris.cache"), parameter(parameter),
      local_target_cloud(new pcXYZ),
      local_target_normals(new pcNormal)
{
  bool recalculation_is_necessary = isRecalculationNecessary();

  if (!recalculation_is_necessary) {
    all_target_cloud = pcXYZ::Ptr(new pcXYZ);
    all_target_normals = pcNormal::Ptr(new pcNormal);
    all_sparse_cloud = pcXYZ::Ptr(new pcXYZ);

    // load cache file
    bool flag1 = (pcl::io::loadPCDFile<pcl::PointXYZ>(cache_cloud_file, *all_target_cloud) != -1);
    bool flag2 = (pcl::io::loadPCDFile<pcl::Normal>(cache_normals_file, *all_target_normals) != -1);
    bool flag3 = (pcl::io::loadPCDFile<pcl::PointXYZ>(cache_sparse_file, *all_sparse_cloud) != -1);

    if (flag1 && flag2 && flag3)
      std::cout << "Because of cache hit, cached target point cloud was loaded" << std::endl;
    else
      recalculation_is_necessary = true;
  }

  if (recalculation_is_necessary) {
    std::cout << "Because of cache miss, recalculate the target point cloud" << std::endl;

    std::cout << "start loading pointcloud & esitmating normal with leafsize " << parameter.voxel_grid_leaf << " search_radius " << parameter.normal_search_radius << std::endl;
    all_target_cloud = pcXYZ::Ptr(new pcXYZ);
    all_target_normals = pcNormal::Ptr(new pcNormal);
    all_sparse_cloud = pcXYZ::Ptr(new pcXYZ);
    util::loadMap(parameter.pcd_file, all_target_cloud, all_target_normals, parameter.voxel_grid_leaf, parameter.normal_search_radius);

    {
      pcl::VoxelGrid<pcl::PointXYZ> filter;
      filter.setInputCloud(all_target_cloud);
      filter.setLeafSize(4 * parameter.voxel_grid_leaf, 4 * parameter.voxel_grid_leaf, 4 * parameter.voxel_grid_leaf);
      filter.filter(*all_sparse_cloud);
    }


    // save as cache file
    std::cout << "save pointcloud" << std::endl;
    pcl::io::savePCDFileBinaryCompressed<pcl::PointXYZ>(cache_cloud_file, *all_target_cloud);
    pcl::io::savePCDFileBinaryCompressed<pcl::Normal>(cache_normals_file, *all_target_normals);
    pcl::io::savePCDFileBinaryCompressed<pcl::PointXYZ>(cache_sparse_file, *all_sparse_cloud);

    // update cache information
    std::ofstream ofs(cache_file);
    ofs << parameter.toString();
  }
  std::cout << "all_target_cloud_size " << all_target_cloud->size() << std::endl;

  // Calculate the number of submap and its size
  std::cout << "It starts making submaps. This may take few seconds." << std::endl;
  float L = parameter.submap_grid_leaf;
  if (L < 1) {
    L = 1;
    std::cout << "please set positive number for parameter.submap_grid_leaf" << std::endl;
  }

  // Make submaps
  for (size_t i = 0; i < all_target_cloud->size(); i++) {
    pcl::PointXYZ p = all_target_cloud->at(i);
    pcl::Normal n = all_target_normals->at(i);

    int id_x = static_cast<int>(std::floor(p.x / L));
    int id_y = static_cast<int>(std::floor(p.y / L));

    std::pair key = std::make_pair(id_x, id_y);
    submap_cloud[key].push_back(p);
    submap_normals[key].push_back(n);
  }

  // Construct local map
  updateLocalmap(T_init);
}

bool Map::isRecalculationNecessary() const
{
  std::ifstream ifs(cache_file);
  // If cahce data doesn't exist, recalculate
  if (!ifs)
    return true;
  std::string data;

  // If cahce data doesn't match with parameter, recalculate
  std::getline(ifs, data);
  if (data != parameter.toString())
    return true;

  return false;
}

bool Map::informCurrentPose(const Eigen::Matrix4f& T)
{
  bool is_necessary = isUpdateNecessary(T);
  if (!is_necessary)
    return false;

  updateLocalmap(T);
  return true;
}

bool Map::isUpdateNecessary(const Eigen::Matrix4f& T) const
{
  // NOTE: The boundaries of the submap have overlaps in order not to vibrate

  // (1) Condition about the location
  float distance = (T.topRightCorner(2, 1) - localmap_info.xy()).cwiseAbs().maxCoeff();
  if (distance > 0.75 * parameter.submap_grid_leaf) {
    std::cout << "map update because of the distance condition" << std::endl;
    return true;
  }

  // (2) Condition about the location
  float yaw = yawFromPose(T);
  if (subtractAngles(yaw, localmap_info.theta) > 60.f / 180.f * 3.14f) {
    std::cout << "map update because of the angle condition" << std::endl;
    return true;
  }


  // Then, it need not to update the localmap
  return false;
}

void Map::updateLocalmap(const Eigen::Matrix4f& T)
{
  std::cout << "\033[1;4;36m###############" << std::endl;
  std::cout << "Update Localmap" << std::endl;
  std::cout << "###############\033[m" << std::endl;

  Eigen::Vector3f t = T.topRightCorner(3, 1);
  const float L = parameter.submap_grid_leaf;
  int id_x = static_cast<int>(std::floor(t.x() / L));
  int id_y = static_cast<int>(std::floor(t.y() / L));
  std::cout << "id_x " << id_x << " id_y " << id_y << std::endl;

  int pattern = static_cast<int>(yawFromPose(T) / (3.14f / 4.0f));
  int x_min, y_min, dx, dy;
  float new_info_theta;
  switch (pattern) {
  case 0:
  case 7:
    x_min = id_x - 1;
    y_min = id_y - 1;
    dx = 4;
    dy = 3;
    new_info_theta = 0;
    break;
  case 1:
  case 2:
    x_min = id_x - 1;
    y_min = id_y - 1;
    dx = 3;
    dy = 4;
    new_info_theta = 3.1415f * 0.5f;
    break;
  case 3:
  case 4:
    x_min = id_x - 2;
    y_min = id_y - 1;
    dx = 4;
    dy = 3;
    new_info_theta = 3.1415f;
    break;
  case 5:
  case 6:
  default:
    x_min = id_x - 1;
    y_min = id_y - 2;
    dx = 3;
    dy = 4;
    new_info_theta = 3.1415f * 1.5f;
    break;
  }

  // Critical section from here
  {
    local_target_cloud->clear();
    local_target_normals->clear();

    for (int i = 0; i < dx; i++) {
      for (int j = 0; j < dy; j++) {
        std::pair<int, int> key = std::make_pair(x_min + i, y_min + j);
        if (submap_cloud.count(key) == 0) {
          continue;
        }
        *local_target_cloud += submap_cloud[key];
        *local_target_normals += submap_normals[key];
      }
    }
  }
  {
    localmap_info.x = (static_cast<float>(id_x) + 0.5f) * L,
    localmap_info.y = (static_cast<float>(id_y) + 0.5f) * L,
    localmap_info.theta = new_info_theta;
  }
  std::cout << "new map-info: "
            << localmap_info.x << ", "
            << localmap_info.y << ", "
            << localmap_info.theta
            << std::endl;
  // Critical section until here
}

float Map::yawFromPose(const Eigen::Matrix4f& T) const
{
  Eigen::Matrix3f R = util::normalizeRotation(T);

  // When the optical axis of the camera is pointing to the X-axis
  // and the upper side of the camera is pointing to the Z-axis,
  // the rotation matrix is as follows,
  Eigen::Matrix3f camera_rotate;
  camera_rotate << 0, 0, 1,
      -1, 0, 0,
      0, -1, 0;

  // Therefore, multiply the inverse rotation matrix of it.
  // To extract the rotation on the XY-plane, we calculate how a unit vector is moved by a remained rotation.
  Eigen::Vector3f direction = (R * camera_rotate.transpose()) * Eigen::Vector3f::UnitX();

  float theta = std::atan2(direction.y(), direction.x());  // [-pi,pi]
  if (theta < 0)
    return theta + 6.28f;
  return theta;
}

}  // namespace map
}  // namespace iris