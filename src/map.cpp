#include "map.hpp"

namespace vllm
{
namespace map
{
Map::Map(const Parameter& parameter) : cache_file("vllm.cache"), parameter(parameter)
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

bool Map::isRecalculationNecessary()
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


}  // namespace map
}  // namespace vllm