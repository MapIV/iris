#include "vllm/system/system.hpp"
#include "vllm/optimize/aligner.hpp"
#include "vllm/optimize/averager.hpp"
#include "vllm/optimize/optimizer.hpp"
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>

namespace vllm
{

System::System(const Config& config_, const std::shared_ptr<map::Map>& map_)
{
  config = config_;
  correspondences = pcl::CorrespondencesPtr(new pcl::Correspondences);
  map = map_;

  // System starts in initialization mode
  vllm_state = VllmState::Inittializing;

  // Setup correspondence estimator
  estimator.setInputTarget(map->getTargetCloud());
  estimator.setTargetNormals(map->getTargetNormals());
  estimator.setKSearch(20);

  localmap_info = map->getLocalmapInfo();

  recollection.store(config.recollection);

  T_world = config.T_init;
  T_align.setIdentity();
  vllm_velocity.setZero();

  // for optimizer module
  optimize::Gain gain;
  gain.scale = config.scale_gain;
  gain.smooth = config.smooth_gain;
  gain.altitude = config.altitude_gain;
  gain.latitude = config.latitude_gain;

  optimize_config.gain = gain;
  optimize_config.distance_min = config.distance_min;
  optimize_config.distance_max = config.distance_max;
  optimize_config.iteration = config.iteration;
  optimize_config.threshold_rotation = config.converge_rotation;
  optimize_config.threshold_translation = config.converge_translation;
  optimize_config.ref_scale = util::getScale(config.T_init);

  // During the constructor funtion, there is no way to access members from other threads
  thread_safe_optimize_gain = optimize_config.gain;

  for (int i = 0; i < history; i++)
    vllm_history.push_front(T_world);
}

int System::execute(int vslam_state, const Eigen::Matrix4f& T_vslam, const pcXYZIN::Ptr& vslam_data)
{
  // ====================
  if (vllm_state == VllmState::Inittializing) {
    // NOTE: "2" means openvslam::tracking_state_t::Tracking
    if (vslam_state == 2) vllm_state = VllmState::Tracking;
    T_align = T_world;
  }

  // =======================
  if (vllm_state == VllmState::Lost) {
    std::cerr << "\033[31mVllmState::Lost has not been implemented yet.\033[m" << std::endl;
    exit(1);
  }

  // ====================
  if (vllm_state == VllmState::Relocalizing) {
    std::cerr << "\033[31mVllmState::Relocalizing has not been implemented yet.\033[m" << std::endl;
    exit(1);
  }

  // ====================
  if (vllm_state == VllmState::Tracking) {
    // Optimization
    updateOptimizeGain();
    optimizer.setConfig(optimize_config);
    Eigen::Matrix4f T_initial_align = T_align;

    std::cout << "T_align\n"
              << T_align << std::endl;

    optimize::Outcome outcome = optimizer.optimize(
        map, vslam_data, T_vslam, estimator, T_initial_align, vllm_history);
    correspondences = outcome.correspondences;
    T_align = outcome.T_align;
  }

  // update the pose in the world
  T_world = T_align * T_vslam;

  // Update local map
  map->informCurrentPose(T_world);
  map::Info new_localmap_info = map->getLocalmapInfo();

  // Reinitialize correspondencesEstimator
  if (localmap_info != new_localmap_info) {
    localmap_info = new_localmap_info;
    correspondences->clear();
    estimator.setInputTarget(map->getTargetCloud());
    estimator.setTargetNormals(map->getTargetNormals());
  }

  // Update history
  vllm_history.pop_back();
  vllm_history.push_front(T_world);

  // Pubush for the viewer
  vllm_trajectory.push_back(T_world.topRightCorner(3, 1));
  offset_trajectory.push_back((config.T_init * T_vslam).topRightCorner(3, 1));
  publisher.push(
      T_align, T_world, config.T_init * T_vslam,
      vslam_data, vllm_trajectory,
      offset_trajectory, correspondences, localmap_info);

  last_T_vslam = T_vslam;

  return vllm_state;
}
}  // namespace vllm