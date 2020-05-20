#include "system/publisher.hpp"

namespace vllm
{
bool Publisher::pop(Publication& p)
{
  std::lock_guard<std::mutex> lock(mtx);

  if (flags[(id + 1) % 2] == false) {
    return false;
  }

  p = publication[(id + 1) % 2];
  flags[(id + 1) % 2] = false;
  return true;
}

// NOTE: There are many redundant copies
void Publisher::push(
    const Eigen::Matrix4f& T_align,
    const Eigen::Matrix4f& vllm_camera,
    const Eigen::Matrix4f& offset_camera,
    const pcXYZIN::Ptr& raw_data,
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

  util::transformXYZINormal(raw_data, tmp.cloud, tmp.normals, T_align);

  {
    std::lock_guard<std::mutex> lock(mtx);
    flags[id] = true;
    id = (id + 1) % 2;
  }
}
}  // namespace vllm