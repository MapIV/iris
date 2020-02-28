#include "aligner.hpp"
#include "bridge.hpp"
#include "config.hpp"
#include "pangolin_viewer.hpp"
#include "rejector_lpd.hpp"
#include "util.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <popl.hpp>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;


int main(int argc, char* argv[])
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
  // create options
  vllm::Config config(config_file_path->value());

  // setup for target(LiDAR) map
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = vllm::loadMapPointCloud(config.pcd_file, config.voxel_grid_leaf);
  pcl::PointCloud<pcl::Normal>::Ptr target_normals = vllm::estimateNormals(target_cloud, config.normal_search_leaf);

  // setup for Rejector
  vllm::GPD gpd(config.gpd_size, target_cloud, config.gpd_gain);
  vllm::CorrespondenceRejectorLpd lpd_rejector(gpd);
  pcl::registration::CorrespondenceRejectorDistance distance_rejector;

  // setup for OpenVSLAM
  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv, config.video_file, config.frame_skip);

  // setup for others
  bool vllm_pause = false;
  Eigen::Matrix4f T_init = config.T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
  std::vector<Eigen::Vector3f> raw_trajectory;
  std::vector<Eigen::Vector3f> vllm_trajectory;
  Eigen::Matrix4f raw_camera = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f vllm_camera = Eigen::Matrix4f::Identity();

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr aligned_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);

  // setup for Viewer
  vllm::PangolinViewer pangolin_viewer;
  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  // == Main Loop ==
  while (true) {

    // Execute vSLAM
    bool success = bridge.execute();
    if (!success)
      break;

    // Get some information of vSLAM
    bridge.getLandmarksAndNormals(source_cloud, source_normals);
    int vslam_state = static_cast<int>(bridge.getState());
    raw_camera = bridge.getCameraPose().inverse().cast<float>();

    // Visualize by OpenCV
    cv::imshow("OpenCV", bridge.getFrame());
    int key = cv::waitKey(10);
    if (key == 'q') break;
    if (key == 'r') {
      T_align = Eigen::Matrix4f::Identity();
      raw_trajectory.clear();
      vllm_trajectory.clear();
    }
    if (key == 'p') vllm_pause = !vllm_pause;
    if (key == 's')
      while (cv::waitKey(0) == 's')
        ;
    // `2` means openvslam::tracking_state_t::Tracking
    if (vslam_state != 2 || source_cloud->empty()) {
      continue;
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

      // Visualize by Pangolin
      pangolin_viewer.clear();
      pangolin_viewer.drawGridLine();
      pangolin_viewer.drawString("VLLM", {1.0f, 1.0f, 0.0f, 3.0f});

      pangolin_viewer.drawPointCloud(aligned_cloud, {1.0f, 1.0f, 0.0f, 2.0f});
      pangolin_viewer.drawPointCloud(target_cloud, {0.6f, 0.6f, 0.6f, 1.0f});
      pangolin_viewer.drawTrajectory(vllm_trajectory, {1.0f, 0.0f, 1.0f, 1.0f});
      pangolin_viewer.drawCamera(vllm_camera, {1.0f, 0.0f, 0.0f, 2.0f});
      pangolin_viewer.drawNormals(aligned_cloud, aligned_normals, {1.0f, 0.0f, 1.0f, 1.0f});
      // pangolin_viewer.drawCorrespondences(aligned_cloud, target_cloud, correspondences, {0.0f, 1.0f, 0.0f, 2.0f});
      // pangolin_viewer.drawCamera(raw_camera, {1.0f, 0.0f, 1.0f, 1.0f});
      // pangolin_viewer.drawTrajectory(raw_trajectory, {1.0f, 0.0f, 1.0f, 3.0f});
      // pangolin_viewer.drawNormals(target_cloud, target_normals, {0.0f, 1.0f, 1.0f, 1.0f}, 30);
      // pangolin_viewer.drawGPD(gpd);
      pangolin_viewer.swap();
    }
    vllm_trajectory.push_back(vllm_camera.block(0, 3, 3, 1));
  }

  return 0;
}