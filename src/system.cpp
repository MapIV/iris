#include "system.hpp"
#include "aligner.hpp"
#include "averager.hpp"
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>

namespace vllm
{
System::System(Config& config, const std::shared_ptr<map::Map>& map) : config(config), map(map)
{
  // setup for OpenVSLAM
  bridge.setup(config);

  // setup correspondence estimator
  localmap_info = map->getLocalmapInfo();
  estimator.setInputTarget(map->getTargetCloud());
  estimator.setTargetNormals(map->getTargetNormals());
  estimator.setKSearch(10);

  camera_velocity.setIdentity();
  T_init = config.T_init;
  parameter.scale_gain = config.scale_gain;
  parameter.smooth_gain = config.smooth_gain;
  parameter.altitude_gain = config.altitude_gain;
  parameter.latitude_gain = config.latitude_gain;
  {
    std::lock_guard<std::mutex> lock(parameter_mutex);
    thread_safe_parameter = parameter;
  }

  search_distance_min = config.distance_min;
  search_distance_max = config.distance_max;

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

  // Artifical restart
  if (reset_requested.load()) {
    reset_requested.store(false);
    T_align = Eigen::Matrix4f::Identity();
    database.vllm_camera = T_init;
    old_vllm_camera = T_init;
    older_vllm_camera = T_init;
  }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    bridge.requestReset();
  }

  // Update alignment parameter
  updateParameter();


  std::cout << "state " << vslam_state << std::endl;
  aligning_mode = (vslam_state == 2);


  if (aligning_mode) {
    least_one = true;

    if (relocalizing) {
      relocalizing = false;
      int period = bridge.getPeriodFromInitialId();

      std::cout << "====== period " << period << " ======" << std::endl;
      std::cout << "camera_velocity\n"
                << camera_velocity << std::endl;

      Eigen::Matrix4f tmp = database.vllm_camera;
      Eigen::Matrix4f inv = camera_velocity.inverse();
      for (int i = 0; i < period; i++) {
        tmp = inv * tmp;
      }

      Eigen::Vector3f dx = (tmp - database.vllm_camera).topRightCorner(3, 1);
      std::cout << "dx " << dx.transpose() << std::endl;
      float drift = std::max(dx.norm(), 0.01f);
      std::cout << "drift " << drift << std::endl;

      Eigen::Matrix3f sR = T_init.topLeftCorner(3, 3);
      T_init.topLeftCorner(3, 3) = drift * sR;
    }
    camera_velocity.setZero();

    // Get keypoints cloud with normals
    pcXYZ::Ptr source_cloud(new pcXYZ);
    pcNormal::Ptr source_normals(new pcNormal);
    bridge.getLandmarksAndNormals(source_cloud, source_normals, recollection, accuracy);

    // Update threshold to adjust the number of points
    if (source_cloud->size() < 400 && accuracy > 0.01)
      accuracy -= 0.01;
    if (source_cloud->size() > 600 && accuracy < 0.99)
      accuracy += 0.01;

    Eigen::Matrix4f vslam_camera = bridge.getCameraPose().inverse().cast<float>();

    // Transform subtract the first pose offset
    database.offset_camera = T_init * vslam_camera;
    pcl::transformPointCloud(*source_cloud, *database.offset_cloud, T_init);
    vllm::transformNormals(*source_normals, *database.offset_normals, T_init);

    // Main Optimization
    for (int i = 0; i < 5; i++) {
      if (optimize(i))
        break;
    }
  }
  // Inertial model
  else {
    if (camera_velocity.isZero()) {
      camera_velocity = calcVelocity(camera_history);
      std::cout << "calc camera_velocity\n"
                << camera_velocity << std::endl;
    }

    database.offset_cloud->clear();
    database.offset_normals->clear();

    T_init = camera_velocity * old_vllm_camera;
    T_align.setIdentity();
    database.offset_camera = T_init;
    database.vllm_camera = T_init;

    // Has initialization been successful at least one?
    if (least_one) {
      database.vllm_camera = vllm::getNormalizedPose(database.vllm_camera);
      relocalizing = true;
    }
  }

  // std::cout << "T_init\n"
  //           << T_init << std::endl;
  // std::cout << "now= " << database.vllm_camera.topRightCorner(3, 1).transpose() << std::endl;

  // Update local map
  map->informCurrentPose(database.vllm_camera);
  map::Info new_localmap_info = map->getLocalmapInfo();
  if (localmap_info != new_localmap_info) {
    // Reinitialize KDtree
    localmap_info = new_localmap_info;
    estimator.setInputTarget(map->getTargetCloud());
    estimator.setTargetNormals(map->getTargetNormals());
  }


  // Copy data for viewer
  database.offset_trajectory.push_back(database.offset_camera.block(0, 3, 3, 1));
  database.vllm_trajectory.push_back(database.vllm_camera.block(0, 3, 3, 1));
  publisher.push(database);

  // Update old data
  older_vllm_camera = old_vllm_camera;
  old_vllm_camera = database.vllm_camera;
  camera_history.pop_back();
  camera_history.push_front(database.vllm_camera);
  return 0;
}

bool System::optimize(int iteration)
{
  std::cout << "itr= \033[32m" << iteration << "\033[m";

  // Integrate
  pcl::transformPointCloud(*database.offset_cloud, *database.vllm_cloud, T_align);
  vllm::transformNormals(*database.offset_normals, *database.vllm_normals, T_align);

  // Get all correspodences
  estimator.setInputSource(database.vllm_cloud);
  estimator.setSourceNormals(database.vllm_normals);
  estimator.determineCorrespondences(*database.correspondences);
  std::cout << " ,raw_correspondences= \033[32m" << database.correspondences->size() << "\033[m";

  // Reject too far correspondences
  distance_rejector.setInputCorrespondences(database.correspondences);
  distance_rejector.setMaximumDistance(
      search_distance_max - (search_distance_max - search_distance_min) * static_cast<float>(iteration) / static_cast<float>(config.iteration));
  distance_rejector.getCorrespondences(*database.correspondences);
  std::cout << " ,refined_correspondecnes= \033[32m" << database.correspondences->size() << "\033[m" << std::endl;

  // Align pointclouds
  vllm::Aligner aligner;
  aligner.setPrePosition(database.offset_camera, old_vllm_camera, older_vllm_camera);
  aligner.setParameter(parameter);
  T_align = aligner.estimate7DoF(T_align, *database.offset_cloud, *map->getTargetCloud(), *database.correspondences, map->getTargetNormals(), database.offset_normals);

  // Integrate
  Eigen::Matrix4f last_camera = database.vllm_camera;
  database.vllm_camera = T_align * database.offset_camera;

  // Get Inovation
  float scale = getScale(getNormalizedRotation(database.vllm_camera));
  float update_transform = (last_camera - database.vllm_camera).topRightCorner(3, 1).norm();        // called "Euclid distance"
  float update_rotation = (last_camera - database.vllm_camera).topLeftCorner(3, 3).norm() / scale;  // called "chordal distance"
  std::cout << "update= \033[33m" << update_transform << " \033[m,\033[33m " << update_rotation << "\033[m" << std::endl;

  if (config.converge_translation > update_transform
      && config.converge_rotation > update_rotation)
    return true;

  return false;
}

}  // namespace vllm