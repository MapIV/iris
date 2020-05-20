#include "optimize/optimizer.hpp"
#include "core/util.hpp"
#include "optimize/aligner.hpp"
#include <iostream>

namespace vllm
{
namespace optimize
{
int method_num = 0;

Outcome Optimizer::optimize(
    const std::shared_ptr<map::Map>& map_ptr,
    const pcXYZIN::Ptr& vslam_data,
    const Eigen::Matrix4f& offset_camera,
    crrspEstimator& estimator,
    const Eigen::Matrix4f& T_initial_align,
    const std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& vllm_history)
{
  pcXYZ::Ptr tmp_cloud(new pcXYZ);
  pcNormal::Ptr tmp_normals(new pcNormal);
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);

  Eigen::Matrix4f T_align = T_initial_align;

  for (int itr = 0; itr < config.iteration; itr++) {
    std::cout << "itration= \033[32m" << itr << "\033[m";

    // Initial transform
    util::transformXYZINormal(vslam_data, tmp_cloud, tmp_normals, T_align);

    Eigen::Vector3f offset_pos = (T_align * offset_camera).topRightCorner(3, 1);

    // TODO: We should enable the estimator handle the PointXYZINormal
    estimator.setInputSource(tmp_cloud);
    estimator.setSourceNormals(tmp_normals);
    estimator.setCenter(offset_pos);
    estimator.setMethod(method_num);
    estimator.determineCorrespondences(*correspondences);
    method_num = (method_num + 1) % 2;

    std::cout << " ,raw_correspondences= \033[32m" << correspondences->size() << "\033[m";

    // NOTE: distance_rejector doesn't seemd to work well.
    // Reject too far correspondences
    float distance = config.distance_max - (config.distance_max - config.distance_min) * static_cast<float>(itr) / static_cast<float>(config.iteration);
    distance_rejector.setInputCorrespondences(correspondences);
    distance_rejector.setMaximumDistance(distance);
    distance_rejector.getCorrespondences(*correspondences);
    std::cout << " ,refined_correspondecnes= \033[32m" << correspondences->size() << "\033[m" << std::endl;

    Eigen::Matrix4f vllm_camera = T_align * offset_camera;
    Eigen::Matrix4f last_camera = vllm_camera;

    // Align pointclouds
    optimize::Aligner aligner(config.gain.scale, config.gain.latitude, config.gain.altitude, config.gain.smooth);
    T_align = aligner.estimate7DoF(
        T_align, vslam_data, map_ptr->getTargetCloud(), correspondences,
        offset_camera, vllm_history, config.ref_scale, map_ptr->getTargetNormals());

    // Integrate
    vllm_camera = T_align * offset_camera;

    // Get Inovation
    float scale = util::getScale(vllm_camera);
    float update_transform = (last_camera - vllm_camera).topRightCorner(3, 1).norm();        // called "Euclid distance"
    float update_rotation = (last_camera - vllm_camera).topLeftCorner(3, 3).norm() / scale;  // called "chordal distance"
    std::cout << "update= \033[33m" << update_transform << " \033[m,\033[33m " << update_rotation << "\033[m" << std::endl;

    if (config.threshold_translation > update_transform
        && config.threshold_rotation > update_rotation)
      break;
  }

  Outcome outcome;
  outcome.correspondences = correspondences;
  outcome.T_align = T_align;
  return outcome;
}


}  // namespace optimize
}  // namespace vllm