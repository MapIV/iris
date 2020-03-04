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
      correspondences(new pcl::Correspondences)
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
}

int System::update()
{
  // Execute vSLAM
  bool success = bridge.execute();
  if (!success)
    return -1;

  // Get some information of vSLAM
  bridge.getLandmarksAndNormals(source_cloud, source_normals);
  vslam_state = static_cast<int>(bridge.getState());
  raw_camera = bridge.getCameraPose().inverse().cast<float>();

  // "2" means openvslam::tracking_state_t::Tracking
  if (vslam_state != 2 || source_cloud->empty()) {
    return -2;
  }
  // "3" means openvslam::tracking_state_t::Lost
  if (vslam_state == 3) {
    std::cout << "\n\033[33m";
    for (int i = 0; i < 10; i++) {
      std::cout << "###########" << std::endl;
      if (i == 5) std::cout << " VSLAM LOST";
    }
    std::cout << "\n\033[m" << std::endl;
  }

  if (reset_requested) {
    reset_requested = false;
    T_align = Eigen::Matrix4f::Identity();
  }

  // Transform subtract the first pose offset
  raw_camera = T_init * raw_camera;
  raw_trajectory.push_back(raw_camera.block(0, 3, 3, 1));
  pcl::transformPointCloud(*source_cloud, *source_cloud, T_init);
  pcl::transformPointCloud(*source_cloud, *aligned_cloud, T_align);
  vllm::transformNormals(*source_normals, *source_normals, T_init);
  vllm::transformNormals(*source_normals, *aligned_normals, T_align);

  if (first_set) {
    first_set = false;
  } else
    vllm_trajectory.push_back(vllm_camera.block(0, 3, 3, 1));

  return 0;
}

std::pair<float, float> System::optimize(int iteration)
{
  std::cout << "itr = \033[32m" << iteration << "\033[m";
  if (source_cloud->empty())
    return {0, 0};

  // Get all correspodences
  // correspondences = vllm::getCorrespondences(aligned_cloud, target_cloud);
  estimator.setInputSource(aligned_cloud);
  estimator.setSourceNormals(aligned_normals);
  estimator.determineCorrespondences(*correspondences);
  std::cout << " ,raw_crsp= \033[32m" << correspondences->size() << "\033[m";

  // Reject enough far correspondences
  distance_rejector.setInputCorrespondences(correspondences);
  distance_rejector.setMaximumDistance(config.distance_max - (config.distance_max - config.distance_min) * static_cast<float>(iteration) / static_cast<float>(config.iteration));
  distance_rejector.getCorrespondences(*correspondences);
  std::cout << " ,rejected by distance= \033[32m" << correspondences->size() << "\033[m" << std::endl;

  // Align pointclouds
  vllm::Aligner aligner;
  aligner.setGain(scale_restriction_gain, pitch_restriction_gain);
  T_align = aligner.estimate7DoF(T_align, *source_cloud, *target_cloud, *correspondences, target_normals, source_normals);

  // Integrate
  vllm_camera = T_align * raw_camera;
  pcl::transformPointCloud(*source_cloud, *aligned_cloud, T_align);
  vllm::transformNormals(*source_normals, *aligned_normals, T_align);

  // Get Inovation
  float scale = getScale(getNormalizedRotation(vllm_camera));
  float update_transform = (last_vllm_camera - vllm_camera).topRightCorner(3, 1).norm();        // called "Euclid distance"
  float update_rotation = (last_vllm_camera - vllm_camera).topLeftCorner(3, 3).norm() / scale;  // called "chordal distance"
  std::cout << "update= \033[33m" << update_transform << " \033[m,\033[33m " << update_rotation << "\033[m" << std::endl;
  last_vllm_camera = vllm_camera;

  return {update_transform, update_rotation};
}

}  // namespace vllm