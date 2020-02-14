#include "aligner.hpp"
#include "bridge.hpp"
#include "pangolin_viewer.hpp"
#include "rejector_lpd.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/correspondence_estimation.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

// L2 norm is used
pcl::Correspondences getCorrespondences(const pcXYZ::Ptr& cloud_source, const pcXYZ::Ptr& cloud_target)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source, target;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
  est.setInputSource(cloud_source);
  est.setInputTarget(cloud_target);
  pcl::Correspondences all_correspondences;
  est.determineCorrespondences(all_correspondences);

  return all_correspondences;
}

float leaf_size = 0.05f;

pcl::PointCloud<pcl::PointXYZ>::Ptr loadMapPointCloud()
{

  // Load map pointcloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::io::loadPCDFile<pcl::PointXYZ>("../data/room.pcd", *cloud_map);
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud_map);
  filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  filter.filter(*cloud_map);
  return cloud_map;
}

Eigen::Matrix4f loadInitialTransform()
{
  cv::FileStorage fs("../data/config.yaml", cv::FileStorage::READ);
  cv::Mat t, r;
  float s;
  fs["VLLM.t_init"] >> t;
  fs["VLLM.r_init"] >> r;
  fs["VLLM.s_init"] >> s;
  fs["VLLM.leaf"] >> leaf_size;

  cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
  cv::Mat T_inv = cv::Mat::eye(4, 4, CV_32FC1);
  cv::Rodrigues(r, r);
  cv::Mat sR = s * r;

  sR.copyTo(T.colRange(0, 3).rowRange(0, 3));
  t.copyTo(T.col(3).rowRange(0, 3));

  Eigen::Matrix4f T_init;
  cv::cv2eigen(T, T_init);
  std::cout << T_init << std::endl;

  return T_init;
}

int main(int argc, char* argv[])
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target = loadMapPointCloud();
  Eigen::Matrix4f T_init = loadInitialTransform();

  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv);

  vllm::PangolinViewer pangolin_viewer;
  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  // rejector with global point distribution
  vllm::GPD gpd(5);
  gpd.init(cloud_target, 0.2f);
  vllm::CorrespondenceRejectorLpd rejector(gpd);

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

    for (int i = 0; i < 20; i++) {
      // Get all correspodences
      pcl::Correspondences correspondences = getCorrespondences(local_cloud, cloud_target);
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
      pangolin_viewer.drawStateString(state);
      pangolin_viewer.drawPointCloud(local_cloud, {1.0f, 1.0f, 0.0f, 2.0f});
      pangolin_viewer.drawPointCloud(global_cloud, {1.0f, 0.0f, 0.0f, 1.0f});
      pangolin_viewer.drawPointCloud(cloud_target, {0.8f, 0.8f, 0.8f, 1.0f});
      pangolin_viewer.drawCamera(camera, {0.0f, 1.0f, 0.0f, 2.0f});
      pangolin_viewer.drawGPD(gpd);
      pangolin_viewer.swap();
    }
  }

  return 0;
}