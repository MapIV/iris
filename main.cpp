#include "bridge.hpp"
#include "pangolin_viewer.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char* argv[])
{
  Eigen::Matrix4d T_init;
  Eigen::Matrix4d T_init_inv;
  {
    cv::FileStorage fs("../data/config.yaml", cv::FileStorage::READ);
    cv::Mat t, r;
    float s;
    fs["VLLM.t_init"] >> t;
    fs["VLLM.r_init"] >> r;
    fs["VLLM.s_init"] >> s;

    cv::Mat T = cv::Mat::ones(4, 4, CV_32FC1);
    cv::Mat T_inv = cv::Mat::ones(4, 4, CV_32FC1);
    cv::Rodrigues(r, r);
    cv::Mat sR = s * r;

    sR.copyTo(T.colRange(0, 3).rowRange(0, 3));
    t.copyTo(T.col(3).rowRange(0, 3));

    cv::Mat(sR.t()).copyTo(T_inv.colRange(0, 3).rowRange(0, 3));
    cv::Mat(-sR.t() * t).copyTo(T_inv.col(3).rowRange(0, 3));

    cv::cv2eigen(T, T_init);
    cv::cv2eigen(T_inv, T_init_inv);
    std::cout << T_init << std::endl;
    std::cout << T_init_inv << std::endl;
  }

  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile<pcl::PointXYZ>("../data/room.pcd", *cloud_map);

  const auto frame_publisher = bridge.get_frame_publisher();
  const auto map_publisher = bridge.get_map_publisher();

  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  PangolinViewer pangolin_viewer;


  while (true) {
    // Execute vSLAM
    bool success = bridge.execute();
    if (!success)
      break;

    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    bridge.getLandmarks(local_cloud, global_cloud);

    int state = static_cast<int>(frame_publisher->get_tracking_state());
    Eigen::Matrix4d camera = bridge.getCameraPose();

    camera = T_init * camera;
    pcl::transformPointCloud(*local_cloud, *local_cloud, T_init_inv);
    pcl::transformPointCloud(*global_cloud, *global_cloud, T_init_inv);

    // Visualize by OpenCV
    cv::imshow("OpenCV", frame_publisher->draw_frame());

    // Visualize by Pangolin
    pangolin_viewer.clear();
    pangolin_viewer.drawGridLine();
    pangolin_viewer.drawState(state);
    pangolin_viewer.addPointCloud(local_cloud, {1.0f, 1.0f, 0.0f, 2.0f});
    pangolin_viewer.addPointCloud(global_cloud, {1.0f, 0.0f, 0.0f, 1.5f});
    pangolin_viewer.addPointCloud(cloud_map, {0.7f, 0.7f, 0.7f, 1.0f});
    pangolin_viewer.addCamera(camera, {0.0f, 1.0f, 0.0f, 2.0f});
    pangolin_viewer.swap();

    int key = cv::waitKey(10);
    if (key == 'q') {
      break;
    }
  }

  return 0;
}