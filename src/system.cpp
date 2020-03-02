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
      source_normals(new pcl::PointCloud<pcl::Normal>)
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

  // setup for Rejector
  gpd.init(config.gpd_size, target_cloud, config.gpd_gain);
  lpd_rejector.init(gpd);

  // setup for OpenVSLAM
  bridge.setup(argc, argv, config.video_file, config.frame_skip);

  T_init = config.T_init;
}

int System::execute()
{
  // Execute vSLAM
  bool success = bridge.execute();
  if (!success)
    return -1;

  // Get some information of vSLAM
  bridge.getLandmarksAndNormals(source_cloud, source_normals);
  int vslam_state = static_cast<int>(bridge.getState());
  raw_camera = bridge.getCameraPose().inverse().cast<float>();

  // `2` means openvslam::tracking_state_t::Tracking
  if (vslam_state != 2 || source_cloud->empty()) {
    return -2;
  }

  // Transform subtract the first pose offset
  raw_camera = T_init * raw_camera;
  raw_trajectory.push_back(raw_camera.block(0, 3, 3, 1));
  pcl::transformPointCloud(*source_cloud, *source_cloud, T_init);
  pcl::transformPointCloud(*source_cloud, *aligned_cloud, T_align);
  vllm::transformNormals(*source_normals, *source_normals, T_init);
  vllm::transformNormals(*source_normals, *aligned_normals, T_align);

  for (int i = 0; i < config.iteration; i++) {
    // Get all correspodences
    pcl::CorrespondencesPtr correspondences = vllm::getCorrespondences(aligned_cloud, target_cloud);
    std::cout << "raw_crsp= \033[32m" << correspondences->size() << "\033[m";

    // Reject enough far correspondences
    distance_rejector.setInputCorrespondences(correspondences);
    distance_rejector.setMaximumDistance(config.distance_max - (config.distance_max - config.distance_min) * static_cast<float>(i) / static_cast<float>(config.iteration));
    distance_rejector.getCorrespondences(*correspondences);
    std::cout << " ,rejected by distance= \033[32m" << correspondences->size() << "\033[m";

    // Reject correspondences don't follow the lpd
    correspondences = lpd_rejector.refineCorrespondences(correspondences, source_cloud);
    std::cout << " ,rejected by lpd= \033[32m" << correspondences->size() << "\033[m" << std::endl;

    // Align pointclouds
    vllm::Aligner aligner;
    aligner.setGain(config.scale_gain, config.pitch_gain);
    if (!vllm_pause)
      T_align = aligner.estimate7DoF(T_align, *source_cloud, *target_cloud, *correspondences, target_normals, source_normals);

    // Integrate
    vllm_camera = T_align * raw_camera;
    pcl::transformPointCloud(*source_cloud, *aligned_cloud, T_align);
    vllm::transformNormals(*source_normals, *aligned_normals, T_align);
  }
  vllm_trajectory.push_back(vllm_camera.block(0, 3, 3, 1));

  return 0;
}


}  // namespace vllm