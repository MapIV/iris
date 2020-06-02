#pragma once
#include "core/keypoints_with_normal.hpp"
#include "core/util.hpp"
#include "map/info.hpp"
#include <Eigen/Dense>
#include <mutex>
#include <vector>

namespace iris
{
struct Publication {
  Publication() : cloud(new pcXYZ), normals(new pcNormal), correspondences(new pcl::Correspondences) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix4f iris_camera;
  Eigen::Matrix4f offset_camera;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> iris_trajectory;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> offset_trajectory;
  map::Info localmap_info;

  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;
  pcl::CorrespondencesPtr correspondences;
};

// thread safe publisher
class Publisher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Publisher()
  {
    flags[0] = flags[1] = false;
    id = 0;
  }

  bool pop(Publication& p);
  void push(
      const Eigen::Matrix4f& T_align,
      const Eigen::Matrix4f& iris_camera,
      const Eigen::Matrix4f& offset_camera,
      const pcXYZIN::Ptr& raw_data,
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& iris_trajectory,
      const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& offset_trajectory,
      const pcl::CorrespondencesPtr& corre,
      const map::Info& localmap_info);

private:
  int id;
  bool flags[2];
  Publication publication[2];
  mutable std::mutex mtx;
};

}  // namespace iris