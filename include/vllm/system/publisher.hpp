#pragma once
#include "vllm/core/keypoints_with_normal.hpp"
#include "vllm/core/util.hpp"
#include <Eigen/Dense>
#include <mutex>
#include <vector>

namespace vllm
{
struct Publication {
  Publication() : cloud(new pcXYZ), normals(new pcNormal), correspondences(new pcl::Correspondences) {}

  Eigen::Matrix4f vllm_camera;
  Eigen::Matrix4f offset_camera;
  std::vector<Eigen::Vector3f> vllm_trajectory;
  std::vector<Eigen::Vector3f> offset_trajectory;
  map::Info localmap_info;

  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;
  pcl::CorrespondencesPtr correspondences;
};

// thread safe publisher
class Publisher
{
private:
  Publication publication[2];
  std::mutex mtx;
  int id = 0;
  bool flag[2] = {false, false};

public:
  // TODO: There are many redundant copies
  void push(
      const Eigen::Matrix4f& T_align,
      const Eigen::Matrix4f& vllm_camera,
      const Eigen::Matrix4f& offset_camera,
      const KeypointsWithNormal& offset,
      const std::vector<Eigen::Vector3f>& vllm_trajectory,
      const std::vector<Eigen::Vector3f>& offset_trajectory,
      const pcl::CorrespondencesPtr& corre,
      const map::Info& localmap_info)
  {
    Publication& tmp = publication[id];

    tmp.vllm_camera = vllm_camera;
    tmp.offset_camera = offset_camera;
    tmp.vllm_trajectory = vllm_trajectory;
    tmp.offset_trajectory = offset_trajectory;
    tmp.localmap_info = localmap_info;

    *tmp.correspondences = *corre;
    pcl::transformPointCloud(*offset.cloud, *tmp.cloud, T_align);
    vllm::transformNormals(*offset.normals, *tmp.normals, T_align);

    {
      std::lock_guard lock(mtx);
      flag[id] = true;
      id = (id + 1) % 2;
    }
  }

  bool pop(Publication& p)
  {
    std::lock_guard lock(mtx);
    if (flag[(id + 1) % 2] == false) {
      return false;
    }

    p = publication[(id + 1) % 2];
    flag[(id + 1) % 2] = false;
    return true;
  }
};
}  // namespace vllm