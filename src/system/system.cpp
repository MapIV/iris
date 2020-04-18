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
  state = State::Inittializing;

  // Setup for OpenVSLAM
  bridge.setup(config);

  // Setup correspondence estimator
  estimator.setInputTarget(map->getTargetCloud());
  estimator.setTargetNormals(map->getTargetNormals());
  estimator.setKSearch(10);

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

int System::execute(const cv::Mat& image)
{
  // Execute vSLAM
  bridge.execute(image);
  int vslam_state = static_cast<int>(bridge.getState());

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

  // ====================
  if (state == State::Inittializing) {

    // "2" means openvslam::tracking_state_t::Tracking
    if (vslam_state == 2) state = State::Tracking;

    // ######################
    T_align = T_world;
    // ######################
  }

  // =======================
  if (state == State::Lost) {
    std::cerr << "\033[31mvllm::Lost has not been implemented yet.\033[m" << std::endl;
    exit(1);

    // // "2" means openvslam::tracking_state_t::Tracking
    // if (vslam_state == 2) state = State::Relocalizing;

    // if (vllm_velocity.isZero()) vllm_velocity = optimize::calcVelocity(vllm_history);
    // Eigen::Matrix4f last_vllm_camera = *vllm_history.begin();

    // // ######################
    // T_align = T_world;
    // // ######################
  }

  // ====================
  if (state == State::Relocalizing) {
    std::cerr << "\033[31mvllm::Relocalization has not been implemented yet.\033[m" << std::endl;
    exit(1);
    // state = State::Tracking;
    // int period = bridge.getPeriodFromInitialId();
    // std::cout << "====== period " << period << " ======" << std::endl;
    // std::cout << "vllm_velocity\n"
    //           << vllm_velocity << std::endl;

    // Eigen::Matrix4f tmp = T_init;
    // Eigen::Matrix4f inv = vllm_velocity.inverse();
    // for (int i = 0; i < period; i++) {
    //   tmp = inv * tmp;
    // }
    // // Reset velocity
    // vllm_velocity.setZero();

    // Eigen::Vector3f dx = (tmp - T_init).topRightCorner(3, 1);
    // float drift = std::max(dx.norm(), 0.01f);
    // std::cout << "drift " << drift << std::endl;

    // Eigen::Matrix3f sR = T_init.topLeftCorner(3, 3);
    // // ######################
    // T_init.topLeftCorner(3, 3) = drift * sR;
    // T_align.setIdentity();
    // // ######################
  }

  // ====================
  if (state == State::Tracking) {
    // Get valid camera pose in vSLAM world
    vslam_camera = bridge.getCameraPose().inverse();

    // Get keypoints cloud with normals
    bridge.setCriteria(30, 0.5);
    bridge.getLandmarksAndNormals(raw_keypoints.cloud, raw_keypoints.normals, weights);

    // Update threshold to adjust the number of points
    if (raw_keypoints.cloud->size() < 300 && accuracy > 0.10) accuracy -= 0.01;
    if (raw_keypoints.cloud->size() > 500 && accuracy < 0.90) accuracy += 0.01;

    // Optimization
    updateOptimizeGain();
    optimizer.setConfig(optimize_config);
    Eigen::Matrix4f T_initial_align = T_align;

    std::cout << "T_align\n"
              << T_align << std::endl;

    if (!T_imu.isZero()) {
      std::cout << "T_imu * (T_vslam)^-1\n"
                << T_imu * (vslam_camera.inverse()) << std::endl;
      T_initial_align = T_imu * vslam_camera.inverse();  // Feedback!!
      T_imu.setZero();
    }

    optimize::Outcome outcome = optimizer.optimize(map, raw_keypoints, vslam_camera, estimator, T_initial_align, vllm_history, weights);

    // Retrieve outcome
    correspondences = outcome.correspondences;
    // ######################
    T_align = outcome.T_align;
    // ######################
  }

  // update the pose in the world
  T_world = T_align * vslam_camera;
  std::cout << "final T_world\n"
            << T_world << std::endl;

  // Update local map
  map->informCurrentPose(T_world);
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
  vllm_history.push_front(T_world);

  // Pubush for the viewer
  vllm_trajectory.push_back(T_world.topRightCorner(3, 1));
  offset_trajectory.push_back((config.T_init * vslam_camera).topRightCorner(3, 1));
  publisher.push(
      T_align, T_world, config.T_init * vslam_camera,
      raw_keypoints, vllm_trajectory,
      offset_trajectory, correspondences, localmap_info);

  return state;
}
}  // namespace vllm