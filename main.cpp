#include "aligner.hpp"
#include "bridge.hpp"
#include "pangolin_viewer.hpp"
#include "rejector_lpd.hpp"
#include "util.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/correspondence_estimation.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

struct Config {
  Config(std::string yaml_file)
  {
    cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
    {
      cv::Mat t, r;
      float s;
      fs["VLLM.t_init"] >> t;
      fs["VLLM.r_init"] >> r;
      fs["VLLM.s_init"] >> s;
      cv::Rodrigues(r, r);
      cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
      cv::Mat(s * r).copyTo(T.colRange(0, 3).rowRange(0, 3));
      t.copyTo(T.col(3).rowRange(0, 3));
      cv::cv2eigen(T, T_init);
      std::cout << T_init << std::endl;
    }

    fs["VLLM.normal_search_leaf"] >> normal_search_leaf;
    fs["VLLM.voxel_grid_leaf"] >> voxel_grid_leaf;
    fs["VLLM.pcd_file"] >> pcd_file;
    fs["VLLM.gpd_size"] >> gpd_size;
    fs["VLLM.gpd_gain"] >> gpd_gain;
    fs["VLLM.iteration"] >> iteration;
    std::cout << "gpd_gain " << gpd_gain << std::endl;

    cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
  }

  int iteration;
  float gpd_gain;
  int gpd_size;
  float normal_search_leaf;
  float voxel_grid_leaf;
  std::string pcd_file;
  Eigen::Matrix4f T_init;
};

int main(int argc, char* argv[])
{
  Config config("../data/config.yaml");

  // setup for LiDAR map
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target = vllm::loadMapPointCloud(config.pcd_file, config.voxel_grid_leaf);
  pcl::PointCloud<pcl::Normal>::Ptr normals = vllm::estimateNormals(cloud_target, config.normal_search_leaf);

  // setup for OpenVSLAM
  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv);

  // setup for Viewer
  vllm::PangolinViewer pangolin_viewer;
  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  // setup for Rejector
  vllm::GPD gpd(config.gpd_size);
  gpd.init(cloud_target, config.gpd_gain);
  vllm::CorrespondenceRejectorLpd rejector(gpd);

  const Eigen::Matrix4f T_init = config.T_init;

  // == Main Loop ==
  while (true) {
    // Execute vSLAM
    bool success = bridge.execute();
    if (!success)
      break;

    // Get some information of vSLAM
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    bridge.getLandmarks(local_cloud, global_cloud);
    int state = static_cast<int>(bridge.getState());
    Eigen::Matrix4f camera = bridge.getCameraPose().inverse().cast<float>();

    // Visualize by OpenCV
    cv::imshow("OpenCV", bridge.getFrame());
    if (cv::waitKey(10) == 'q') break;

    // `2` means openvslam::tracking_state_t::Tracking
    if (state != 2 || local_cloud->empty()) {
      continue;
    }

    // Transform to subtract initial offset
    camera = T_init * camera;
    pcl::transformPointCloud(*local_cloud, *local_cloud, T_init);
    pcl::transformPointCloud(*global_cloud, *global_cloud, T_init);

    for (int i = 0; i < config.iteration; i++) {
      // Get all correspodences
      pcl::Correspondences correspondences = vllm::getCorrespondences(local_cloud, cloud_target);
      // Reject invalid correspondeces
      correspondences = rejector.refineCorrespondences(correspondences, local_cloud);
      // Align pointclouds
      vllm::Aligner aligner;
      Eigen::Matrix4f T;
      aligner.estimate(*local_cloud, *cloud_target, correspondences, T);

      camera = T * camera;
      pcl::transformPointCloud(*local_cloud, *local_cloud, T);
      pcl::transformPointCloud(*global_cloud, *global_cloud, T);

      // Visualize by Pangolin
      pangolin_viewer.clear();
      pangolin_viewer.drawGridLine();
      pangolin_viewer.drawString("state=" + std::to_string(state) + ", itr=" + std::to_string(i), {1.0f, 0.0f, 0.0f, 2.0f});
      pangolin_viewer.drawPointCloud(local_cloud, {1.0f, 1.0f, 0.0f, 2.0f});
      pangolin_viewer.drawPointCloud(global_cloud, {1.0f, 0.0f, 0.0f, 1.0f});
      pangolin_viewer.drawPointCloud(cloud_target, {0.8f, 0.8f, 0.8f, 1.0f});
      pangolin_viewer.drawCorrespondences(
          local_cloud, cloud_target,
          correspondences, {0.0f, 0.8f, 0.0f, 1.0f});
      pangolin_viewer.drawCamera(camera, {0.0f, 1.0f, 0.0f, 1.0f});
      // pangolin_viewer.drawGPD(gpd);
      pangolin_viewer.drawNormals(cloud_target, normals, {0.0f, 0.0f, 1.0f, 2.0f});
      pangolin_viewer.swap();
    }
  }

  return 0;
}