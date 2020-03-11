#include "system.hpp"
#include "aligner.hpp"
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>
#include <popl.hpp>

namespace vllm
{
System::System(int argc, char* argv[])
    : aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>),
      aligned_normals(new pcl::PointCloud<pcl::Normal>),
      source_cloud(new pcl::PointCloud<pcl::PointXYZ>),
      source_normals(new pcl::PointCloud<pcl::Normal>),
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
}

int System::execute()
{
  // Execute vSLAM
  bool success = bridge.execute();
  if (!success)
    return -1;

  // Get some information of vSLAM
  bridge.getLandmarksAndNormals(source_cloud, source_normals, recollection, accuracy);

  int vslam_state = static_cast<int>(bridge.getState());
  if (vslam_state == 2)
    raw_camera = bridge.getCameraPose().inverse().cast<float>() * offset_camera;
  else
    raw_camera = offset_camera;


  if (reset_requested.load()) {
    reset_requested.store(false);
    vslam_state = 3;
  }

  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m";
    for (int i = 0; i < 10; i++) {
      std::cout << "###########" << std::endl;
      if (i == 5) std::cout << " Request Reset";
    }
    std::cout << "\n\033[m" << std::endl;
    bridge.requestReset();
    offset_camera = raw_camera;
    scale_restriction_gain = 0;
    model_restriction_gain = 100;
  }

  // reset estimated transform
  // if (reset_requested) {
  //   reset_requested = false;
  //   T_align = Eigen::Matrix4f::Identity();
  // }

  // update threshold to adjust the number of points
  if (source_cloud->size() < 400 && accuracy > 0.01)
    accuracy -= 0.01;
  if (source_cloud->size() > 600 && accuracy < 0.99)
    accuracy += 0.01;


  // Transform subtract the first pose offset
  raw_camera = T_init * raw_camera;
  pcl::transformPointCloud(*source_cloud, *source_cloud, T_init);
  vllm::transformNormals(*source_normals, *source_normals, T_init);

  // Main Optimization
  for (int i = 0; i < 5; i++) {
    if (optimize(i))
      break;
  }

  {
    // Copy data for viewer
    std::lock_guard<std::mutex> lock(mtx);
    vllm_camera = T_align * raw_camera;
    std::cout << vllm_camera.topRightCorner(3, 1).transpose() << " " << pre_camera.transpose() << " " << pre_pre_camera.transpose() << std::endl;

    pcl::transformPointCloud(*source_cloud, *aligned_cloud, T_align);
    vllm::transformNormals(*source_normals, *aligned_normals, T_align);

    raw_trajectory.push_back(raw_camera.block(0, 3, 3, 1));
    vllm_trajectory.push_back(vllm_camera.block(0, 3, 3, 1));
    *correspondences_for_viewer = *correspondences;

    pre_pre_camera = pre_camera;
    pre_camera = vllm_camera.topRightCorner(3, 1);
  }

  return 0;
}

bool System::optimize(int iteration)
{
  std::cout << "itr= \033[32m" << iteration << "\033[m";

  // Integrate
  pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr tmp_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::transformPointCloud(*source_cloud, *tmp_cloud, T_align);
  vllm::transformNormals(*source_normals, *tmp_normals, T_align);

  // Get all correspodences
  estimator.setInputSource(tmp_cloud);
  estimator.setSourceNormals(tmp_normals);
  estimator.determineCorrespondences(*correspondences);
  std::cout << " ,raw_crsp= \033[32m" << correspondences->size() << "\033[m";

  // Reject enough far correspondences
  distance_rejector.setInputCorrespondences(correspondences);
  distance_rejector.setMaximumDistance(
      search_distance_max - (search_distance_max - search_distance_min) * static_cast<float>(iteration) / static_cast<float>(config.iteration));
  distance_rejector.getCorrespondences(*correspondences);
  std::cout << " ,rejected by distance= \033[32m" << correspondences->size() << "\033[m" << std::endl;

  // Align pointclouds
  vllm::Aligner aligner;
  aligner.setPrePosition(raw_camera.topRightCorner(3, 1), pre_camera, pre_pre_camera);
  aligner.setGain(scale_restriction_gain, pitch_restriction_gain, model_restriction_gain);
  T_align = aligner.estimate7DoF(T_align, *source_cloud, *target_cloud, *correspondences, target_normals, source_normals);

  // Integrate
  Eigen::Matrix4f now_camera = T_align * raw_camera;

  // Get Inovation
  float scale = getScale(getNormalizedRotation(now_camera));
  float update_transform = (last_vllm_camera - now_camera).topRightCorner(3, 1).norm();        // called "Euclid distance"
  float update_rotation = (last_vllm_camera - now_camera).topLeftCorner(3, 3).norm() / scale;  // called "chordal distance"
  std::cout << "update= \033[33m" << update_transform << " \033[m,\033[33m " << update_rotation << "\033[m" << std::endl;
  last_last_vllm_camera = last_vllm_camera;
  last_vllm_camera = now_camera;

  if (config.converge_translation > update_transform
      && config.converge_rotation > update_rotation)
    return true;

  return false;
}

}  // namespace vllm