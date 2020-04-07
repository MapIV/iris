#include "vllm/map/map.hpp"
#include "vllm/core/util.hpp"
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

    std::cout << "start loading pointcloud & esitmating normal with leafsize " << parameter.voxel_grid_leaf << " search_radius " << parameter.normal_search_radius << std::endl;
    all_target_cloud = pcXYZ::Ptr(new pcXYZ);
    all_target_normals = pcNormal::Ptr(new pcNormal);
    vllm::loadMap(parameter.pcd_file, all_target_cloud, all_target_normals, parameter.voxel_grid_leaf, parameter.normal_search_radius);

    // save as cache file
    std::cout << "save pointcloud with normal" << parameter.normal_search_radius << std::endl;
    pcl::io::savePCDFileBinaryCompressed<pcl::PointXYZ>(cache_cloud_file, *all_target_cloud);
    pcl::io::savePCDFileBinaryCompressed<pcl::Normal>(cache_normals_file, *all_target_normals);

    // update cache information
    std::ofstream ofs(cache_file);
    ofs << parameter.toString();
  }
  std::cout << "all_target_cloud_size " << all_target_cloud->size() << std::endl;

  // Calculate the number of submap and its size
  std::cout << "It starts making submaps. This may take few seconds." << std::endl;
  const float L = parameter.submap_grid_leaf;
  pcl::PointXYZ minimum, maximum;
  pcl::getMinMax3D(*all_target_cloud, minimum, maximum);
  max_corner_point = maximum.getArray3fMap();
  min_corner_point = minimum.getArray3fMap();

  grid_x_num = static_cast<int>((maximum.x - minimum.x) / L) + 1;
  grid_y_num = static_cast<int>((maximum.y - minimum.y) / L) + 1;
  if (grid_x_num < 3) grid_x_num = 3;
  if (grid_y_num < 3) grid_y_num = 3;

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

      std::vector<int> indices;
      crop.setMin(min4);
      crop.setMax(max4);
      crop.filter(indices);

      pcXYZ cropped_cloud;
      pcNormal cropped_normals;
      pcl::copyPointCloud(*all_target_cloud, indices, cropped_cloud);
      pcl::copyPointCloud(*all_target_normals, indices, cropped_normals);

      submap_cloud.push_back(std::move(cropped_cloud));
      submap_normals.push_back(std::move(cropped_normals));
    }
  }

  // Construct local map
  updateLocalmap(Eigen::Matrix4f::Identity());
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
  // std::cout << "distance-condition: " << distance
  //           << " self " << T.topRightCorner(2, 1).transpose()
  //           << " info" << localmap_info.xy().transpose() << std::endl;
  if (distance > 0.75 * parameter.submap_grid_leaf) {
    return true;
  }

  // (2) Condition about the location
  float yaw = yawFromPose(T);
  // std::cout << "angle-condition: " << yaw << " " << localmap_info.theta << std::endl;
  if (subtractAngles(yaw, localmap_info.theta) > 60.f / 180.f * 3.14f) {
    return true;
  }

  // Then, it need not to update the localmap
  return false;
}

void Map::updateLocalmap(const Eigen::Matrix4f& T)
{
  std::cout << "###############" << std::endl;
  std::cout << "Update Localmap" << std::endl;
  std::cout << "###############" << std::endl;

  Eigen::Vector3f dP = (T.topRightCorner(3, 1) - min_corner_point);
  const float L = parameter.submap_grid_leaf;
  int cx = static_cast<int>(dP.x() / L);
  int cy = static_cast<int>(dP.y() / L);
  std::cout << "cx " << cx << " cy " << cy << std::endl;

  // TODO:
  int pattern = static_cast<int>(yawFromPose(T) / (3.14f / 4.0f));
  // int pattern = 0;
  int x_min, y_min, dx, dy;
  float new_info_theta;
  switch (pattern) {
  case 0:
  case 7:
    x_min = cx - 1;
    y_min = cy - 1;
    dx = 4;
    dy = 3;
    new_info_theta = 0;
    break;
  case 1:
  case 2:
    x_min = cx - 1;
    y_min = cy - 1;
    dx = 3;
    dy = 4;
    new_info_theta = 3.1415f * 0.5f;
    break;
  case 3:
  case 4:
    x_min = cx - 2;
    y_min = cy - 1;
    dx = 4;
    dy = 3;
    new_info_theta = 3.1415f;
    break;
  case 5:
  case 6:
  default:
    x_min = cx - 1;
    y_min = cy - 2;
    dx = 3;
    dy = 4;
    new_info_theta = 3.1415f * 1.5f;
    break;
  }
  std::cout << "pattern " << pattern << " " << x_min << " " << y_min << std::endl;

  // Critical section from here
  {
    std::lock_guard lock(localmap_mtx);
    local_target_cloud->clear();
    local_target_normals->clear();

    for (int i = 0; i < dx; i++) {
      if (i + x_min < 0) continue;
      if (i + x_min == grid_x_num) continue;

      for (int j = 0; j < dy; j++) {
        if (j + y_min < 0) continue;
        if (j + y_min == grid_y_num) continue;

        int tmp = (j + y_min) + grid_y_num * (i + x_min);
        *local_target_cloud += submap_cloud.at(tmp);
        *local_target_normals += submap_normals.at(tmp);
      }
    }
  }
  {
    std::lock_guard lock(info_mtx);
    localmap_info.x = min_corner_point.x() + (cx + 0.5f) * L,
    localmap_info.y = min_corner_point.y() + (cy + 0.5f) * L,
    localmap_info.theta = new_info_theta;
  }
  std::cout << "new-info"
            << localmap_info.x << " "
            << localmap_info.y << " "
            << localmap_info.theta << " ,min "
            << min_corner_point.transpose() << " L="
            << L << std::endl;
  // Critical section until here
}

float Map::yawFromPose(const Eigen::Matrix4f& T) const
{
  Eigen::Matrix3f R = normalizeRotation(T);

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
}  // namespace vllm