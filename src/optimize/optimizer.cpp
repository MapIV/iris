#include "optimize/optimizer.hpp"
#include "core/util.hpp"
#include "optimize/aligner.hpp"
#include <iostream>

namespace vllm
{
namespace optimize
{
Outcome Optimizer::optimize(
    const std::shared_ptr<map::Map>& map_ptr,
    const KeypointsWithNormal& keypoints,  // offset
    const Eigen::Matrix4f& offset_camera,
    crrspEstimator& estimator,
    const Eigen::Matrix4f& T_initial_align
    /*, const std::list<Eigen::Matrix4f>& vllm_histroty*/)
{
  pcXYZ::Ptr tmp_cloud(new pcXYZ);
  pcNormal::Ptr tmp_normals(new pcNormal);
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);

  Eigen::Matrix4f T_align = T_initial_align;

  for (int itr = 0; itr < config.iteration; itr++) {
    std::cout << "itration= \033[32m" << itr << "\033[m";

    // Initial transform
    pcl::transformPointCloud(*keypoints.cloud, *tmp_cloud, T_initial_align);
    vllm::transformNormals(*keypoints.normals, *tmp_normals, T_initial_align);

    // Get all correspodences
    estimator.setInputSource(tmp_cloud);
    estimator.setSourceNormals(tmp_normals);
    estimator.determineCorrespondences(*correspondences);
    std::cout << " ,raw_correspondences= \033[32m" << correspondences->size() << "\033[m";

    // // Reject too far correspondences
    // float distance = config.distance_max - (config.distance_max - config.distance_min) * static_cast<float>(itr) / static_cast<float>(config.iteration);
    // distance_rejector.setInputCorrespondences(correspondences);
    // distance_rejector.setMaximumDistance(distance);
    // distance_rejector.getCorrespondences(*correspondences);
    // std::cout << " ,refined_correspondecnes= \033[32m" << correspondences->size() << "\033[m" << std::endl;

    Eigen::Matrix4f vllm_camera = T_align * offset_camera;
    Eigen::Matrix4f last_camera = vllm_camera;

    // Align pointclouds
    std::cout << "Gain: sclae= " << config.gain.scale << " latitude= " << config.gain.latitude << std::endl;
    optimize::Aligner aligner(config.gain.scale, config.gain.latitude, config.gain.altitude, 0);
    // TODO:
    // aligner.setPrePosition(offset_camera, old_vllm_camera, older_vllm_camera);
    T_align = aligner.estimate7DoF(T_align, *keypoints.cloud, *map_ptr->getTargetCloud(), *correspondences, map_ptr->getTargetNormals(), keypoints.normals);

    // Integrate
    vllm_camera = T_align * offset_camera;

    // Get Inovation
    float scale = getScale(getNormalizedRotation(vllm_camera));
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