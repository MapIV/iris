#pragma once
#include <mutex>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcNormal = pcl::PointCloud<pcl::Normal>;

struct Database {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcXYZ::Ptr offset_cloud;
  pcXYZ::Ptr vllm_cloud;
  pcNormal::Ptr offset_normals;
  pcNormal::Ptr vllm_normals;
  Eigen::Matrix4f offset_camera;
  Eigen::Matrix4f vllm_camera;
  std::vector<Eigen::Vector3f> offset_trajectory;
  std::vector<Eigen::Vector3f> vllm_trajectory;
  pcl::CorrespondencesPtr correspondences;

  Database() : offset_cloud(new pcXYZ),
               vllm_cloud(new pcXYZ),
               offset_normals(new pcNormal),
               vllm_normals(new pcNormal),
               correspondences(new pcl::Correspondences)
  {
    offset_camera.setIdentity();
    vllm_camera.setIdentity();
  }

  // overload
  Database operator=(const Database& other)
  {
    // Deep Copy
    *this->offset_cloud = *other.offset_cloud;
    *this->vllm_cloud = *other.vllm_cloud;
    *this->offset_normals = *other.offset_normals;
    *this->vllm_normals = *other.vllm_normals;
    this->offset_camera = other.offset_camera;
    this->vllm_camera = other.vllm_camera;
    this->offset_trajectory = other.offset_trajectory;
    this->vllm_trajectory = other.vllm_trajectory;
    *this->correspondences = *other.correspondences;

    return *this;
  }
};

// thread safe publisher
class Publisher
{
private:
  Database pub[2];
  std::mutex mtx;
  int id = 0;
  bool flag[2] = {false, false};

public:
  void push(const Database& p)
  {
    pub[id] = p;
    flag[id] = true;

    std::lock_guard lock(mtx);
    id = (id + 1) % 2;
  }

  bool pop(Database& p)
  {
    std::lock_guard lock(mtx);
    if (flag[(id + 1) % 2] == false) {
      return false;
    }

    p = pub[(id + 1) % 2];
    flag[(id + 1) % 2] = false;
    return true;
  }
};
}  // namespace vllm