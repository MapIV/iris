#include "vllm/system/publisher.hpp"

namespace vllm
{
bool Publisher::pop(Publication& p)
{
  std::cout << "try to pop " << std::boolalpha << flags[0] << " " << flags[1] << " " << flags << std::endl;
  // std::lock_guard<std::mutex> lock(mtx);

  if (flags[(id + 1) % 2] == false) {
    return false;
  }

  // p = publication[(id + 1) % 2];
  flags[(id + 1) % 2] = false;
  return true;
}

// NOTE: There are many redundant copies
void Publisher::push(
    const Eigen::Matrix4f& T_align,
    const Eigen::Matrix4f& vllm_camera,
    const Eigen::Matrix4f& offset_camera,
    const KeypointsWithNormal& raw_keypoints,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& vllm_trajectory,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& offset_trajectory,
    const pcl::CorrespondencesPtr& corre,
    const map::Info& localmap_info)
{
  Publication& tmp = publication[id];

  tmp.vllm_camera = util::normalizePose(vllm_camera);
  tmp.offset_camera = util::normalizePose(offset_camera);
  tmp.vllm_trajectory = vllm_trajectory;
  tmp.offset_trajectory = offset_trajectory;
  tmp.localmap_info = localmap_info;
  *tmp.correspondences = *corre;

  pcl::transformPointCloud(*raw_keypoints.cloud, *tmp.cloud, T_align);
  util::transformNormals(*raw_keypoints.normals, *tmp.normals, T_align);

  {
    std::lock_guard<std::mutex> lock(mtx);
    flags[id] = true;
    id = (id + 1) % 2;
  }
  std::cout << "sytem::publisher pushed " << std::boolalpha << flags[0] << " " << flags[1] << " " << flags << std::endl;
}
}  // namespace vllm