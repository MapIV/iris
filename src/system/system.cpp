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
  localmap_info = map->getLocalmapInfo();
  // setup for OpenVSLAM
  bridge.setup(config);

  // setup correspondence estimator
  estimator.setInputTarget(map->getTargetCloud());
  estimator.setTargetNormals(map->getTargetNormals());
  estimator.setKSearch(10);

  camera_velocity.setIdentity();
  T_init = config.T_init;
  T_align.setIdentity();

  //  for optimizer module
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

  for (int i = 0; i < history; i++)
    vllm_history.push_front(T_init);
}

bool least_one = false;

int System::execute()
{
  // Execute vSLAM
  bool success = bridge.execute();
  if (!success)
    return -1;

  int vslam_state = static_cast<int>(bridge.getState());
  Eigen::Matrix4f vslam_camera = Eigen::Matrix4f::Identity();

  // TODO:
  // Artifical restart
  // if (reset_requested.load()) {
  //   reset_requested.store(false);
  //   T_align = Eigen::Matrix4f::Identity();
  //   old_vllm_camera = T_init;
  // }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    bridge.requestReset();
  }

  std::cout << "vslam state " << vslam_state << std::endl;
  aligning_mode = (vslam_state == 2);

  KeypointsWithNormal raw_keypoints;
  KeypointsWithNormal offset_keypoints;

  // std::cout << "T_align\n"
  //           << T_align << std::endl;
  // std::cout << "T_init\n"
  //           << T_init << std::endl;

  if (aligning_mode) {
    least_one = true;

    if (relocalizing) {
      relocalizing = false;
      int period = bridge.getPeriodFromInitialId();

      std::cout << "====== period " << period << " ======" << std::endl;
      std::cout << "camera_velocity\n"
                << camera_velocity << std::endl;

      Eigen::Matrix4f tmp = T_init;
      Eigen::Matrix4f inv = camera_velocity.inverse();
      for (int i = 0; i < period; i++) {
        tmp = inv * tmp;
      }

      Eigen::Vector3f dx = (tmp - T_init).topRightCorner(3, 1);
      std::cout << "dx " << dx.transpose() << std::endl;
      float drift = std::max(dx.norm(), 0.01f);
      std::cout << "drift " << drift << std::endl;

      Eigen::Matrix3f sR = T_init.topLeftCorner(3, 3);
      T_init.topLeftCorner(3, 3) = drift * sR;
    }
    camera_velocity.setZero();

    // Get keypoints cloud with normals
    bridge.getLandmarksAndNormals(raw_keypoints.cloud, raw_keypoints.normals, recollection, accuracy);

    // Update threshold to adjust the number of points
    if (raw_keypoints.cloud->size() < 400 && accuracy > 0.01) accuracy -= 0.01;
    if (raw_keypoints.cloud->size() > 600 && accuracy < 0.99) accuracy += 0.01;

    // Get valid camera pose in vSLAM world
    vslam_camera = bridge.getCameraPose().inverse().cast<float>();

    // Transform subtract the first pose offset
    pcl::transformPointCloud(*raw_keypoints.cloud, *offset_keypoints.cloud, T_init);
    vllm::transformNormals(*raw_keypoints.normals, *offset_keypoints.normals, T_init);

    // Update prameter of optimize
    updateOptimizeGain();

    // =======================
    // == Main Optimization ==
    optimizer.setConfig(optimize_config);
    optimize::Outcome outcome = optimizer.optimize(map, offset_keypoints, T_init * vslam_camera, estimator, T_align, vllm_history);

    correspondences = outcome.correspondences;
    T_align = outcome.T_align;
    // =======================

  }
  // =======================
  // Inertial model
  else {
    if (camera_velocity.isZero()) {
      camera_velocity = optimize::calcVelocity(vllm_history);
      std::cout << "calc camera_velocity\n"
                << camera_velocity << std::endl;
    }

    Eigen::Matrix4f old_vllm_camera = *vllm_history.begin();
    T_init = camera_velocity * old_vllm_camera;
    T_align.setIdentity();

    // Has initialization been successful at least one?
    if (least_one) {
      relocalizing = true;
    }
  }
  // =======================

  Eigen::Matrix4f offset_camera = T_init * vslam_camera;
  Eigen::Matrix4f vllm_camera = T_align * offset_camera;

  // Update local map
  map->informCurrentPose(vllm_camera);
  map::Info new_localmap_info = map->getLocalmapInfo();
  if (localmap_info != new_localmap_info) {
    // Reinitialize KDtree
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
      offset_keypoints, vllm_trajectory, offset_trajectory, correspondences, localmap_info);

  return 0;
}

}  // namespace vllm