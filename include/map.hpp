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

struct Anchor {
  float x;
  float y;
  float theta;
  static constexpr float epsilon = 1e-6f;

  Anchor() {}
  Anchor(float x, float y, float theta) : x(x), y(y), theta(theta) {}

  Eigen::Vector2f xy() const { return Eigen::Vector2f(x, y); }

  std::string toString() const
  {
    return std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(theta);
  }
  bool isEqual(const Anchor& a, const Anchor& b) const
  {
    if (std::fabs(a.x - b.x) > epsilon)
      return false;
    if (std::fabs(a.y - b.y) > epsilon)
      return false;
    if (std::fabs(a.theta - b.theta) > epsilon)
      return false;
    return true;
  }

  bool operator==(const Anchor& other) const
  {
    return isEqual(*this, other);
  }
  bool operator!=(const Anchor& other) const
  {
    return !isEqual(*this, other);
  }
};

class Map
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit Map(const Parameter& parameter);

  // If the map updates then return true.
  bool informCurrentPose(const Eigen::Matrix4f& T);

  // This informs viewer of whether local map updated or not
  Anchor getLocalmapInfo() const
  {
    std::lock_guard lock(anchor_mtx);
    return localmap_anchor;
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
  Anchor localmap_anchor;

  mutable std::mutex localmap_mtx;
  mutable std::mutex anchor_mtx;


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