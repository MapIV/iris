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
  Map(const Parameter& parameter) : cache_file("vllm.cache"), parameter(parameter)
  {
    bool recalculation_is_necessary = isRecalculationNecessary();

    if (!recalculation_is_necessary) {
      target_cloud = pcXYZ::Ptr(new pcXYZ);
      target_normals = pcNormal::Ptr(new pcNormal);

      // load cache file
      bool flag1 = (pcl::io::loadPCDFile<pcl::PointXYZ>(cache_cloud_file, *target_cloud) != -1);
      bool flag2 = (pcl::io::loadPCDFile<pcl::Normal>(cache_normals_file, *target_normals) != -1);

      if (flag1 && flag2)
        std::cout << "cache is used" << std::endl;
      else
        recalculation_is_necessary = true;
    }

    if (recalculation_is_necessary) {
      std::cout << "cache is not used" << std::endl;

      target_cloud = vllm::loadMapPointCloud(parameter.pcd_file, parameter.voxel_grid_leaf);
      target_normals = vllm::estimateNormals(target_cloud, parameter.normal_search_radius);

      // save as cache file
      pcl::io::savePCDFileBinaryCompressed<pcl::PointXYZ>(cache_cloud_file, *target_cloud);
      pcl::io::savePCDFileBinaryCompressed<pcl::Normal>(cache_normals_file, *target_normals);

      // update cache information
      std::ofstream ofs(cache_file);
      ofs << parameter.toString();
    }

    std::cout << "target_cloud_size " << target_cloud->size() << std::endl;
  }

  const pcXYZ::Ptr getTargetCloud() const { return target_cloud; }
  const pcNormal::Ptr getTargetNormals() const { return target_normals; }

private:
  const std::string cache_file;
  const Parameter parameter;

  pcXYZ::Ptr target_cloud;
  pcNormal::Ptr target_normals;

  const std::string cache_cloud_file = "vllm_cloud.pcd";
  const std::string cache_normals_file = "vllm_normals.pcd";

  bool isRecalculationNecessary()
  {
    std::ifstream ifs(cache_file);
    // If cahce information doesn't exist, recalculate
    if (!ifs)
      return true;
    std::string data;

    // If cahce information doesn't match with parameter, recalculate
    std::getline(ifs, data);
    if (data != parameter.toString())
      return true;

    return false;
  }
};
}  // namespace map
}  // namespace vllm