#pragma once
#include "core/keypoints_with_normal.hpp"
#include "core/util.hpp"
#include <Eigen/Dense>
#include <mutex>
#include <vector>

namespace vllm
{
struct Publication {
  Eigen::Matrix4f vllm_camera;
  Eigen::Matrix4f offset_camera;
  std::vector<Eigen::Matrix4f> vllm_cameras;
  std::vector<Eigen::Matrix4f> offset_cameras;
  pcl::CorrespondencesPtr correspondences;
  map::Info localmap_info;

  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;
  Publication() : cloud(new pcXYZ), normals(new pcNormal) {}
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
      const std::vector<Eigen::Matrix4f>& vllm_cameras,
      const std::vector<Eigen::Matrix4f>& offset_cameras,
      const pcl::CorrespondencesPtr& corre,
      const map::Info& localmap_info)
  {
    Publication& tmp = publication[id];

    tmp.vllm_camera = vllm_camera;
    tmp.vllm_cameras = vllm_cameras;
    tmp.offset_camera = offset_camera;
    tmp.offset_cameras = offset_cameras;
    tmp.correspondences = corre;
    tmp.localmap_info = localmap_info;
    pcl::transformPointCloud(*offset.cloud, *tmp.cloud, T_align);
    vllm::transformNormals(*offset.normals, *tmp.normals, T_align);

    flag[id] = true;

    std::lock_guard lock(mtx);
    id = (id + 1) % 2;
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