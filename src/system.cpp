#include "system.hpp"
#include "aligner.hpp"
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
}


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
    vslam_state = 3;
  }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    bridge.requestReset();
  }

  // reset database
  source_cloud->clear();
  source_normals->clear();

  std::cout << "state " << vslam_state << std::endl;
  aligning_mode = (vslam_state == 2);

  if (aligning_mode) {
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
    // Assumes the scale does not change
    double scale = vllm::getScale(old_vllm_camera.topLeftCorner(3, 3));
    Eigen::Matrix3f R1 = vllm::getNormalizedRotation(old_vllm_camera);
    Eigen::Matrix3f R2 = vllm::getNormalizedRotation(older_vllm_camera);
    Eigen::Vector3f t1 = old_vllm_camera.topRightCorner(3, 1);
    Eigen::Vector3f t2 = older_vllm_camera.topRightCorner(3, 1);

    T_init.topLeftCorner(3, 3) = scale * (R1 * R2.transpose() * R1);
    T_init.topRightCorner(3, 1) = 2 * t1 - t2;

    double scale3 = vllm::getScale((R1 * R2.transpose() * R1));
    double scale2 = vllm::getScale(T_init.topLeftCorner(3, 3));

    std::cout << "T_init: " << scale << " " << scale2 << " " << scale3 << "\n"
              << T_init << std::endl;

    T_align.setIdentity();
    offset_camera = T_init;
    vllm_camera = T_init;
  }

  // std::cout << "now= " << vllm_camera.topRightCorner(3, 1).transpose()
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