#pragma once
#include <sstream>

namespace iris
{
namespace map
{
struct Parameter {
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

}  // namespace map
}  // namespace iris