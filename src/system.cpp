#include "system.hpp"
#include "aligner.hpp"
#include "averager.hpp"
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>
#include <popl.hpp>

namespace vllm
{
System::System(int argc, char* argv[])
    : view_vllm_cloud(new pcl::PointCloud<pcl::PointXYZ>),
      view_vllm_normals(new pcl::PointCloud<pcl::Normal>),
      source_cloud(new pcl::PointCloud<pcl::PointXYZ>),
      source_normals(new pcl::PointCloud<pcl::Normal>),
      offset_cloud(new pcl::PointCloud<pcl::PointXYZ>),
      offset_normals(new pcl::PointCloud<pcl::Normal>),
      correspondences(new pcl::Correspondences),
      correspondences_for_viewer(new pcl::Correspondences)
{
  // analyze arugments
  popl::OptionParser op("Allowed options");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!config_file_path->is_set()) {
    std::cerr << "invalid arguments" << std::endl;
    exit(EXIT_FAILURE);
  }
  config.init(config_file_path->value());

  // setup for target(LiDAR) map
  target_cloud = vllm::loadMapPointCloud(config.pcd_file, config.voxel_grid_leaf);
  target_normals = vllm::estimateNormals(target_cloud, config.normal_search_leaf);

  // setup for OpenVSLAM
  bridge.setup(argc, argv, config.video_file, config.frame_skip);

  // setup correspondence estimator
  estimator.setInputTarget(target_cloud);
  estimator.setTargetNormals(target_normals);
  estimator.setKSearch(10);

  T_init = config.T_init;
  scale_restriction_gain = config.scale_gain;
  pitch_restriction_gain = config.pitch_gain;
  model_restriction_gain = config.model_gain;
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
    vllm_camera = T_init;
    vllm_camera = T_init;
    old_vllm_camera = T_init;
    older_vllm_camera = T_init;
  }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    bridge.requestReset();
  }


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

      Eigen::Matrix4f tmp = vllm_camera;
      Eigen::Matrix4f inv = camera_velocity.inverse();
      for (int i = 0; i < period; i++) {
        tmp = inv * tmp;
      }

      Eigen::Vector3f dx = (tmp - vllm_camera).topRightCorner(3, 1);
      std::cout << "dx " << dx.transpose() << std::endl;
      float drift = std::max(dx.norm(), 0.01f);
      std::cout << "drift " << drift << std::endl;

      Eigen::Matrix3f sR = T_init.topLeftCorner(3, 3);
      T_init.topLeftCorner(3, 3) = drift * sR;
    }
    camera_velocity.setZero();

    bridge.getLandmarksAndNormals(source_cloud, source_normals, recollection, accuracy);

    // update threshold to adjust the number of points
    if (source_cloud->size() < 400 && accuracy > 0.01)
      accuracy -= 0.01;
    if (source_cloud->size() > 600 && accuracy < 0.99)
      accuracy += 0.01;

    Eigen::Matrix4f vslam_camera = bridge.getCameraPose().inverse().cast<float>();

    // Transform subtract the first pose offset
    offset_camera = T_init * vslam_camera;
    pcl::transformPointCloud(*source_cloud, *offset_cloud, T_init);
    vllm::transformNormals(*source_normals, *offset_normals, T_init);

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

    offset_cloud->clear();
    offset_normals->clear();

    T_init = camera_velocity * old_vllm_camera;
    T_align.setIdentity();
    offset_camera = T_init;
    vllm_camera = T_init;

    if (least_one) {
      vllm_camera = vllm::getNormalizedPose(vllm_camera);
      relocalizing = true;
    }
  }

  std::cout << "T_init\n"
            << T_init << std::endl;

  std::cout << "now= " << vllm_camera.topRightCorner(3, 1).transpose() << std::endl;
  //           << " pre=" << old_vllm_camera.topRightCorner(3, 1).transpose()
  //           << " prepre=" << older_vllm_camera.topRightCorner(3, 1).transpose() << std::endl;

  // Copy data for viewer
  {
    std::lock_guard<std::mutex> lock(mtx);

    pcl::transformPointCloud(*offset_cloud, *view_vllm_cloud, T_align);
    vllm::transformNormals(*offset_normals, *view_vllm_normals, T_align);

    offset_trajectory.push_back(offset_camera.block(0, 3, 3, 1));
    vllm_trajectory.push_back(vllm_camera.block(0, 3, 3, 1));
    *correspondences_for_viewer = *correspondences;
    view_vllm_camera = vllm_camera;
    view_offset_camera = offset_camera;
  }

  // update old data
  older_vllm_camera = old_vllm_camera;
  old_vllm_camera = vllm_camera;
  camera_history.pop_back();
  camera_history.push_front(vllm_camera);
  return 0;
}


bool System::optimize(int iteration)
{
  // std::cout << "offset= " << offset_camera.topRightCorner(3, 1).transpose() << std::endl;
  std::cout << "itr= \033[32m" << iteration << "\033[m";

  // Integrate
  pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr tmp_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::transformPointCloud(*offset_cloud, *tmp_cloud, T_align);
  vllm::transformNormals(*offset_normals, *tmp_normals, T_align);

  // Get all correspodences
  estimator.setInputSource(tmp_cloud);
  estimator.setSourceNormals(tmp_normals);
  estimator.determineCorrespondences(*correspondences);
  std::cout << " ,raw_correspondences= \033[32m" << correspondences->size() << "\033[m";

  // Reject too far correspondences
  distance_rejector.setInputCorrespondences(correspondences);
  distance_rejector.setMaximumDistance(
      search_distance_max - (search_distance_max - search_distance_min) * static_cast<float>(iteration) / static_cast<float>(config.iteration));
  distance_rejector.getCorrespondences(*correspondences);
  std::cout << " ,refined_correspondecnes= \033[32m" << correspondences->size() << "\033[m" << std::endl;

  // Align pointclouds
  vllm::Aligner aligner;
  aligner.setPrePosition(offset_camera, old_vllm_camera, older_vllm_camera);
  aligner.setGain(scale_restriction_gain, pitch_restriction_gain, model_restriction_gain);
  T_align = aligner.estimate7DoF(T_align, *offset_cloud, *target_cloud, *correspondences, target_normals, offset_normals);

  // Integrate
  Eigen::Matrix4f last_camera = vllm_camera;
  vllm_camera = T_align * offset_camera;

  // Get Inovation
  float scale = getScale(getNormalizedRotation(vllm_camera));
  float update_transform = (last_camera - vllm_camera).topRightCorner(3, 1).norm();        // called "Euclid distance"
  float update_rotation = (last_camera - vllm_camera).topLeftCorner(3, 3).norm() / scale;  // called "chordal distance"
  std::cout << "update= \033[33m" << update_transform << " \033[m,\033[33m " << update_rotation << "\033[m" << std::endl;

  if (config.converge_translation > update_transform
      && config.converge_rotation > update_rotation)
    return true;

  return false;
}

}  // namespace vllm