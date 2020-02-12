#include "bridge.hpp"
#include "pangolin_viewer.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char* argv[])
{
  Eigen::Matrix4d T_init;
  float leaf_size = 0.1f;
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
    cv::cv2eigen(T, T_init);
    std::cout << T_init << std::endl;
  }

  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv);

  // Load map pointcloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>);
  {
    pcl::io::loadPCDFile<pcl::PointXYZ>("../data/room.pcd", *cloud_map);
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud_map);
    filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    filter.filter(*cloud_map);
  }

  PangolinViewer pangolin_viewer;
  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

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
    Eigen::Matrix4d camera = bridge.getCameraPose().inverse();

    // Transform to subtract initial offset
    camera = T_init * camera;
    pcl::transformPointCloud(*local_cloud, *local_cloud, T_init);
    pcl::transformPointCloud(*global_cloud, *global_cloud, T_init);

    // Visualize by OpenCV
    cv::imshow("OpenCV", bridge.getFrame());

    // Visualize by Pangolin
    pangolin_viewer.clear();
    pangolin_viewer.drawGridLine();
    pangolin_viewer.drawState(state);
    pangolin_viewer.addPointCloud(local_cloud, {1.0f, 1.0f, 0.0f, 2.0f});
    pangolin_viewer.addPointCloud(global_cloud, {1.0f, 0.0f, 0.0f, 1.0f});
    pangolin_viewer.addPointCloud(cloud_map, {0.8f, 0.8f, 0.8f, 1.0f});
    pangolin_viewer.addCamera(camera, {0.0f, 1.0f, 0.0f, 2.0f});
    pangolin_viewer.swap();

    // Wait
    int key = cv::waitKey(10);
    if (key == 's') {
      while (key != 'r') {
        key = cv::waitKey();
      }
    }
    if (key == 'q') {
      break;
    }
  }

  return 0;
}