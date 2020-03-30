#include "system/system.hpp"
#include "optimize/aligner.hpp"
#include "optimize/averager.hpp"
#include "optimize/optimizer.hpp"
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>

namespace vllm
{
System::System(Config& config, const std::shared_ptr<map::Map>& map)
    : config(config), map(map), correspondences(new pcl::Correspondences)
{
  // System starts in initialization mode
  state = State::Inittializing;

  // Setup for OpenVSLAM
  bridge.setup(config);

  // Setup correspondence estimator
  estimator.setInputTarget(map->getTargetCloud());
  estimator.setTargetNormals(map->getTargetNormals());
  estimator.setKSearch(40);

  localmap_info = map->getLocalmapInfo();

  recollection.store(config.recollection);

  T_init = config.T_init;
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

  // During the constructor funtion, there is no way to access members from other threads
  thread_safe_optimize_gain = optimize_config.gain;

  for (int i = 0; i < history; i++)
    vllm_history.push_front(T_init);
}

int System::execute(const cv::Mat& image)
{
  // Execute vSLAM
  bridge.execute(image);
  std::cout << "openvslam execute successfully" << std::endl;
  int vslam_state = static_cast<int>(bridge.getState());
  Eigen::Matrix4f vslam_camera = Eigen::Matrix4f::Identity();

  // Artifical reset
  if (reset_requested.load()) {
    reset_requested.store(false);
    T_align = Eigen::Matrix4f::Identity();
  }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    bridge.requestReset();
    state = State::Lost;
  }

  KeypointsWithNormal raw_keypoints;
  KeypointsWithNormal offset_keypoints;

  // ====================
  if (state == State::Inittializing) {

    // "2" means openvslam::tracking_state_t::Tracking
    if (vslam_state == 2) state = State::Tracking;

    // ######################
    // T_init doesn't change
    // T_align doesn't change
    // ######################
  }

  // =======================
  if (state == State::Lost) {

    // "2" means openvslam::tracking_state_t::Tracking
    if (vslam_state == 2) state = State::Relocalizing;

    if (vllm_velocity.isZero()) vllm_velocity = optimize::calcVelocity(vllm_history);
    Eigen::Matrix4f last_vllm_camera = *vllm_history.begin();

    // ######################
    T_init = vllm_velocity * last_vllm_camera;
    T_align.setIdentity();
    // ######################
  }

  // ====================
  if (state == State::Relocalizing) {
    state = State::Tracking;

    int period = bridge.getPeriodFromInitialId();
    std::cout << "====== period " << period << " ======" << std::endl;
    std::cout << "vllm_velocity\n"
              << vllm_velocity << std::endl;

    Eigen::Matrix4f tmp = T_init;
    Eigen::Matrix4f inv = vllm_velocity.inverse();
    for (int i = 0; i < period; i++) {
      tmp = inv * tmp;
    }
    // Reset velocity
    vllm_velocity.setZero();

    Eigen::Vector3f dx = (tmp - T_init).topRightCorner(3, 1);
    float drift = std::max(dx.norm(), 0.01f);
    std::cout << "drift " << drift << std::endl;

    Eigen::Matrix3f sR = T_init.topLeftCorner(3, 3);
    // ######################
    T_init.topLeftCorner(3, 3) = drift * sR;
    T_align.setIdentity();
    // ######################
  }

  // ====================
  if (state == State::Tracking) {
    // Get valid camera pose in vSLAM world
    vslam_camera = bridge.getCameraPose().inverse().cast<float>();

    // Get keypoints cloud with normals
    bridge.getLandmarksAndNormals(raw_keypoints.cloud, raw_keypoints.normals, weights, recollection.load(), accuracy);

    // Update threshold to adjust the number of points
    if (raw_keypoints.cloud->size() < 300 && accuracy > 0.10) accuracy -= 0.01;
    if (raw_keypoints.cloud->size() > 500 && accuracy < 0.90) accuracy += 0.01;

    // Transform subtract the first pose offset
    pcl::transformPointCloud(*raw_keypoints.cloud, *offset_keypoints.cloud, T_init);
    vllm::transformNormals(*raw_keypoints.normals, *offset_keypoints.normals, T_init);

    // Optimization
    updateOptimizeGain();
    optimizer.setConfig(optimize_config);
    optimize::Outcome outcome = optimizer.optimize(map, offset_keypoints, T_init * vslam_camera, estimator, T_align, vllm_history, weights);

    // Retrieve outcome
    correspondences = outcome.correspondences;
    // ######################
    // T_init doesn't change
    T_align = outcome.T_align;
    // ######################
  }

  //  std::cout << "T_init\n" << T_init << std::endl;
  //  std::cout << "T_align\n" << T_align << std::endl;

  // offset_camera =           T_init * vslam_camera
  // vllm_camera   = T_align * T_init * vslam_camera
  Eigen::Matrix4f offset_camera = T_init * vslam_camera;
  Eigen::Matrix4f vllm_camera = T_align * offset_camera;

  T_output = vllm_camera;

  // Update local map
  map->informCurrentPose(vllm_camera);
  map::Info new_localmap_info = map->getLocalmapInfo();
  if (localmap_info != new_localmap_info) {
    // Reinitialize correspondencesEstimator
    localmap_info = new_localmap_info;
    correspondences->clear();
    estimator.setInputTarget(map->getTargetCloud());
    estimator.setTargetNormals(map->getTargetNormals());
  }

  // Update history
  vllm_history.pop_back();
  vllm_history.push_front(vllm_camera);

  // Pubush for the viewer
  vllm_trajectory.push_back(vllm_camera.topRightCorner(3, 1));
  offset_trajectory.push_back(offset_camera.topRightCorner(3, 1));
  publisher.push(
      T_align, vllm_camera, offset_camera,
      offset_keypoints, vllm_trajectory,
      offset_trajectory, correspondences, localmap_info);

  return 0;
}
}  // namespace vllm