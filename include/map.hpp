#pragma once
#include "types.hpp"
#include "util.hpp"
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <sstream>

namespace vllm
{
namespace map
{
struct Parameter {
  Parameter() {}

  Parameter(const std::string pcd_file, float leaf, float radius)
      : pcd_file(pcd_file),
        voxel_grid_leaf(leaf),
        normal_search_radius(radius) {}

  std::string pcd_file;
  float voxel_grid_leaf;
  float normal_search_radius;

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
  Map(const Parameter& parameter);

  const pcXYZ::Ptr getTargetCloud() const { return target_cloud; }
  const pcNormal::Ptr getTargetNormals() const { return target_normals; }

private:
  const std::string cache_file;
  const Parameter parameter;

  pcXYZ::Ptr target_cloud;
  pcNormal::Ptr target_normals;

  const std::string cache_cloud_file = "vllm_cloud.pcd";
  const std::string cache_normals_file = "vllm_normals.pcd";

  bool isRecalculationNecessary();
};
}  // namespace map
}  // namespace vllm