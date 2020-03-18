#include "map.hpp"
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>

namespace vllm
{
namespace map
{
Map::Map(const Parameter& parameter)
    : cache_file("vllm.cache"), parameter(parameter),
      local_target_cloud(new pcXYZ),
      local_target_normals(new pcNormal)
{
  bool recalculation_is_necessary = isRecalculationNecessary();

  if (!recalculation_is_necessary) {
    all_target_cloud = pcXYZ::Ptr(new pcXYZ);
    all_target_normals = pcNormal::Ptr(new pcNormal);

    // load cache file
    bool flag1 = (pcl::io::loadPCDFile<pcl::PointXYZ>(cache_cloud_file, *all_target_cloud) != -1);
    bool flag2 = (pcl::io::loadPCDFile<pcl::Normal>(cache_normals_file, *all_target_normals) != -1);

    if (flag1 && flag2)
      std::cout << "Because of cache hit, cached target point cloud was loaded" << std::endl;
    else
      recalculation_is_necessary = true;
  }

  if (recalculation_is_necessary) {
    std::cout << "Because of cache miss, recalculate the target point cloud" << std::endl;

    all_target_cloud = vllm::loadMapPointCloud(parameter.pcd_file, parameter.voxel_grid_leaf);
    std::cout << parameter.normal_search_radius << std::endl;
    all_target_normals = vllm::estimateNormals(all_target_cloud, parameter.normal_search_radius);

    // save as cache file
    pcl::io::savePCDFileBinaryCompressed<pcl::PointXYZ>(cache_cloud_file, *all_target_cloud);
    pcl::io::savePCDFileBinaryCompressed<pcl::Normal>(cache_normals_file, *all_target_normals);

    // update cache information
    std::ofstream ofs(cache_file);
    ofs << parameter.toString();
  }
  std::cout << "all_target_cloud_size " << all_target_cloud->size() << std::endl;

  // Calculate the number of submap and its size
  const float L = parameter.submap_grid_leaf;
  pcl::PointXYZ minimum, maximum;
  pcl::getMinMax3D(*all_target_cloud, minimum, maximum);
  max_corner_point = maximum.getArray3fMap();
  min_corner_point = minimum.getArray3fMap();

  grid_x_num = (maximum.x - minimum.x) / L + 1.0f;
  grid_y_num = (maximum.y - minimum.y) / L + 1.0f;
  grid_box_unit << L, L, maximum.z - minimum.z;

  pcl::CropBox<pcl::PointXYZ> crop;
  crop.setInputCloud(all_target_cloud);

  Eigen::Vector3f Ly, Lx;
  Lx << L, 0, 0;
  Ly << 0, L, 0;

  // Make submaps
  for (int i = 0; i < grid_x_num; i++) {
    for (int j = 0; j < grid_y_num; j++) {

      Eigen::Vector4f min4 = Eigen::Vector4f::Ones();
      Eigen::Vector4f max4 = Eigen::Vector4f::Ones();
      min4.topRows(3) = min_corner_point + i * Lx + j * Ly;
      max4.topRows(3) = min4.topRows(4) + grid_box_unit;

      pcXYZ cropped_cloud;
      pcNormal cropped_normals;
      crop.setMin(min4);
      crop.setMax(max4);
      crop.filter(cropped_cloud);

      pcl::IndicesPtr indice_ptr = crop.getIndices();
      pcl::copyPointCloud(*all_target_normals, *indice_ptr, cropped_normals);

      submap_cloud.push_back(cropped_cloud);
      submap_normals.push_back(cropped_normals);
    }
  }

  // Construct local map
  update(Eigen::Vector3f::Zero());
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

bool Map::updateLocalMap(const Eigen::Vector3f& pos)
{
  if (!isUpdateNecessary(pos))
    return false;

  update(pos);
  return true;
}

bool Map::isUpdateNecessary(const Eigen::Vector3f& pos)
{
  float dx = (pos - last_grid_center).cwiseAbs().maxCoeff();
  std::cout << "now" << pos.transpose() << "  last " << last_grid_center.transpose() << "  dx" << dx << std::endl;
  // The boundaries of the submap have overlaps in order not to vibrate
  if (dx > 1.25f * parameter.submap_grid_leaf)
    return true;

  return false;
}

void Map::update(const Eigen::Vector3f& pos)
{
  std::cout << "UPDATE SUBMAP" << std::endl;

  // std::lock_guard lock(mtx);

  Eigen::Vector3f dP = (pos - min_corner_point);

  const float L = parameter.submap_grid_leaf;
  int cx = static_cast<int>(dP.x() / L);
  int cy = static_cast<int>(dP.y() / L);
  std::cout << cx << " " << cy << std::endl;
  std::cout << "submap size " << submap_cloud.size() << std::endl;

  local_target_cloud->clear();
  local_target_normals->clear();

  // until here
  // cx \in [0, grid_x_num-1]
  // cy \in [0, grid_y_num-1]
  if (cx < 1) cx += 1;
  if (cy < 1) cy += 1;
  if (cx == grid_x_num - 1) cx -= 1;
  if (cy == grid_y_num - 1) cy -= 1;
  // from here
  // cx \in [1, grid_x_num-2]
  // cy \in [1, grid_y_num-2]

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      int tmp = (j + cy) + grid_y_num * (i + cx);
      std::cout << "tmp " << tmp << " " << i << "," << j << std::endl;
      *local_target_cloud += submap_cloud.at(tmp);
      *local_target_normals += submap_normals.at(tmp);
    }
  }

  std::cout << "local_target_cloud_size " << local_target_cloud->size() << std::endl;

  // update last center
  last_grid_center << min_corner_point.x() + (cx + 0.5f) * L, min_corner_point.y() + (cy + 0.5f) * L, 0;
}

}  // namespace map
}  // namespace vllm