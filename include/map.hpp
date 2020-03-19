#pragma once
#include "types.hpp"
#include "util.hpp"
#include <atomic>
#include <fstream>
#include <mutex>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <sstream>

namespace vllm
{
namespace map
{
struct Parameter {
  Parameter() {}

  Parameter(
      const std::string& pcd_file,
      float voxel_grid_leaf,
      float normal_search_radius,
      float submap_grid_leaf)
      : pcd_file(pcd_file),
        voxel_grid_leaf(voxel_grid_leaf),
        normal_search_radius(normal_search_radius),
        submap_grid_leaf(submap_grid_leaf) {}

  std::string pcd_file;
  float voxel_grid_leaf;
  float normal_search_radius;
  float submap_grid_leaf;

  std::string toString() const
  {
    std::stringstream ss;
    ss << pcd_file << " " << std::to_string(voxel_grid_leaf) << " " << std::to_string(normal_search_radius);
    return ss.str();
  }
};

class Map
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit Map(const Parameter& parameter);

  // If the map updates then return true.
  bool updateLocalMap(const Eigen::Vector3f& pos);

  // TODO: This function may have conflicts
  const pcXYZ::Ptr getTargetCloud() const
  {
    std::lock_guard lock(mtx);
    return local_target_cloud;
  }
  // TODO: This function may have conflicts
  const pcNormal::Ptr getTargetNormals() const
  {
    std::lock_guard lock(mtx);
    return local_target_normals;
  }

  // This informs viewer of whether local map updated or not
  int getLocalmapInfo() const
  {
    return localmap_info.load();
  }

private:
  const std::string cache_file;
  const Parameter parameter;

  const std::string cache_cloud_file = "vllm_cloud.pcd";
  const std::string cache_normals_file = "vllm_normals.pcd";

  // whole point cloud (too heaby)
  pcXYZ::Ptr all_target_cloud;
  pcNormal::Ptr all_target_normals;

  // valid point cloud
  pcXYZ::Ptr local_target_cloud;
  pcNormal::Ptr local_target_normals;

  std::vector<pcXYZ> submap_cloud;
  std::vector<pcNormal> submap_normals;

  Eigen::Vector3f last_grid_center;

  mutable std::mutex mtx;
  std::atomic<int> localmap_info;
  int grid_x_num;
  int grid_y_num;

  Eigen::Vector3f min_corner_point;
  Eigen::Vector3f max_corner_point;
  Eigen::Vector3f grid_box_unit;

  bool isRecalculationNecessary();
  bool isUpdateNecessary(const Eigen::Vector3f& pos);
  void update(const Eigen::Vector3f& pos);
};
}  // namespace map
}  // namespace vllm