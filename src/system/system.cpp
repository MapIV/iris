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


  old_vllm_camera = T_init;
  older_vllm_camera = T_init;

  for (int i = 0; i < history; i++)
    camera_history.push_front(T_init);
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

  // Artifical restart
  if (reset_requested.load()) {
    reset_requested.store(false);
    T_align = Eigen::Matrix4f::Identity();
    old_vllm_camera = T_init;
    older_vllm_camera = T_init;
  }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    bridge.requestReset();
  }

  // TODO:
  // Update alignment parameter
  // updateParameter();

  std::cout << "vslam state " << vslam_state << std::endl;
  aligning_mode = (vslam_state == 2);

  KeypointsWithNormal keypoints;
  KeypointsWithNormal offset;

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
    bridge.getLandmarksAndNormals(keypoints.cloud, keypoints.normals, recollection, accuracy);

    // Update threshold to adjust the number of points
    if (keypoints.cloud->size() < 400 && accuracy > 0.01) accuracy -= 0.01;
    if (keypoints.cloud->size() > 600 && accuracy < 0.99) accuracy += 0.01;

    // Get valid camera pose in vSLAM world
    vslam_camera = bridge.getCameraPose().inverse().cast<float>();

    // Transform subtract the first pose offset
    pcl::transformPointCloud(*keypoints.cloud, *offset.cloud, T_init);
    vllm::transformNormals(*keypoints.normals, *offset.normals, T_init);

    // == Main Optimization ==
    optimizer.setConfig(optimize_config);
    optimize::Outcome outcome
        = optimizer.optimize(map, offset, T_init * vslam_camera, estimator, T_align);
    correspondences = outcome.correspondences;  // TODO: shallow copy?
    T_align = outcome.T_align;
    // == Main Optimization ==

  }
  // Inertial model
  else {
    if (camera_velocity.isZero()) {
      camera_velocity = optimize::calcVelocity(camera_history);
      std::cout << "calc camera_velocity\n"
                << camera_velocity << std::endl;
    }

    T_init = camera_velocity * old_vllm_camera;
    T_align.setIdentity();

    // Has initialization been successful at least one?
    if (least_one) {
      relocalizing = true;
    }
  }

  Eigen::Matrix4f vllm_camera = T_align * T_init * vslam_camera;

  // std::cout << "T_init\n"
  //           << T_init << std::endl;
  // std::cout << "now= " << database.vllm_camera.topRightCorner(3, 1).transpose() << std::endl;

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

  // Pubush for the viewer
  vllm_cameras.push_back(vllm_camera);
  offset_cameras.push_back(T_init * vslam_camera);
  publisher.push(
      T_align, vllm_camera,
      T_init * vslam_camera,
      offset, vllm_cameras, offset_cameras, correspondences, localmap_info);

  // Update old data
  older_vllm_camera = old_vllm_camera;
  old_vllm_camera = vllm_camera;
  camera_history.pop_back();
  camera_history.push_front(vllm_camera);


  return 0;
}

}  // namespace vllm