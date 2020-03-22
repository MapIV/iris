#pragma once
#include "core/types.hpp"

namespace vllm
{
struct KeypointsWithNormal {
  pcXYZ::Ptr cloud;
  pcNormal::Ptr normals;

  KeypointsWithNormal() : cloud(new pcXYZ),
                          normals(new pcNormal)
  {
  }
};
// struct Database {
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//   pcXYZ::Ptr offset_cloud;
//   pcXYZ::Ptr vllm_cloud;
//   pcNormal::Ptr offset_normals;
//   pcNormal::Ptr vllm_normals;
//   Eigen::Matrix4f offset_camera;
//   Eigen::Matrix4f vllm_camera;
//   std::vector<Eigen::Vector3f> offset_trajectory;
//   std::vector<Eigen::Vector3f> vllm_trajectory;
//   pcl::CorrespondencesPtr correspondences;
//   vllm::map::Info localmap_info;

//   Database() : offset_cloud(new pcXYZ),
//                vllm_cloud(new pcXYZ),
//                offset_normals(new pcNormal),
//                vllm_normals(new pcNormal),
//                correspondences(new pcl::Correspondences)
//   {
//     offset_camera.setIdentity();
//     vllm_camera.setIdentity();
//   }

//   // overload
//   Database operator=(const Database& other)
//   {
//     // Deep Copy
//     *this->offset_cloud = *other.offset_cloud;
//     *this->vllm_cloud = *other.vllm_cloud;
//     *this->offset_normals = *other.offset_normals;
//     *this->vllm_normals = *other.vllm_normals;
//     this->offset_camera = other.offset_camera;
//     this->vllm_camera = other.vllm_camera;
//     this->offset_trajectory = other.offset_trajectory;
//     this->vllm_trajectory = other.vllm_trajectory;
//     *this->correspondences = *other.correspondences;
//     this->localmap_info = other.localmap_info;

//     return *this;
//   }
}  // namespace vllm