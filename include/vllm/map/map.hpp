#pragma once
#include "vllm/core/types.hpp"
#include "vllm/core/util.hpp"
#include "vllm/map/info.hpp"
#include "vllm/map/parameter.hpp"
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
class Map
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit Map(const Parameter& parameter);

  // If the map updates then return true.
  bool informCurrentPose(const Eigen::Matrix4f& T);

  // This informs viewer of whether local map updated or not
  Info getLocalmapInfo() const
  {
    std::lock_guard lock(info_mtx);
    return localmap_info;
  }

  // TODO: This function may have conflicts
  const pcXYZ::Ptr getTargetCloud() const
  {
    std::lock_guard lock(localmap_mtx);
    return local_target_cloud;
  }

  // TODO: This function may have conflicts
  const pcNormal::Ptr getTargetNormals() const
  {
    std::lock_guard lock(localmap_mtx);
    return local_target_normals;
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

  // divided point cloud
  std::vector<pcXYZ> submap_cloud;
  std::vector<pcNormal> submap_normals;

  // [x,y,theta]
  Eigen::Vector3f last_grid_center;
  Info localmap_info;

  mutable std::mutex localmap_mtx;
  mutable std::mutex info_mtx;


  int grid_x_num;
  int grid_y_num;

  Eigen::Vector3f min_corner_point;
  Eigen::Vector3f max_corner_point;
  Eigen::Vector3f grid_box_unit;

  bool isRecalculationNecessary() const;
  bool isUpdateNecessary(const Eigen::Matrix4f& T) const;
  void updateLocalmap(const Eigen::Matrix4f& T);

  // return [0,2*pi]
  float yawFromPose(const Eigen::Matrix4f& T) const;

  // return [0,pi]
  float subtractAngles(float a, float b) const
  {
    // a,b \in [0,2\pi]
    float d = std::fabs(a - b);
    if (d > 3.14159f)
      return 2.f * 3.14159f - d;
    return d;
  }
};
}  // namespace map
}  // namespace vllm