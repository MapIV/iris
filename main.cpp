#include "bridge.hpp"
#include "openvslam/data/landmark.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr showLandmarks(
    const std::vector<openvslam::data::landmark*>& landmarks,
    const std::set<openvslam::data::landmark*>& local_landmarks)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (landmarks.empty()) {
    return cloud;
  }

  for (const auto lm : landmarks) {
    if (!lm || lm->will_be_erased()) {
      continue;
    }
    if (local_landmarks.count(lm)) {
      continue;
    }
    const openvslam::Vec3_t pos = lm->get_pos_in_world();
    pcl::PointXYZ p(pos.x(), pos.y(), pos.z());
    cloud->push_back(p);
  }
  for (const auto local_lm : local_landmarks) {
    if (local_lm->will_be_erased()) {
      continue;
    }
    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    pcl::PointXYZ p(pos.x(), pos.y(), pos.z());
    cloud->push_back(p);
  }

  return cloud;
}


int main(int argc, char* argv[])
{
  // #ifdef USE_PANGOLIN_VIEWER
  //   pangolin_viewer::viewer viewer(cfg, &*SLAM_ptr,
  //       SLAM_ptr->get_frame_publisher(), SLAM_ptr->get_map_publisher());
  // #endif

  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv);

  const auto frame_publisher = bridge.get_frame_publisher();
  const auto map_publisher = bridge.get_map_publisher();

  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  bool flag = true;
  while (flag) {
    flag = bridge.execute();
    cv::imshow("OpenCV", frame_publisher->draw_frame());

    int key = cv::waitKey(10);
    if (key == 'q') {
      break;
    }
  }

  // while (!viewer->wasStopped()) {
  //   cv::imshow("OpenCV", frame_publisher->draw_frame());
  //   cv::waitKey(10);

  //   std::vector<openvslam::data::landmark*> landmarks;
  //   std::set<openvslam::data::landmark*> local_landmarks;
  //   map_publisher->get_landmarks(landmarks, local_landmarks);

  //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = showLandmarks(landmarks, local_landmarks);
  //   if (cloud->size() > 0) {
  //     viewer->removeAllPointClouds();
  //     viewer->addPointCloud(cloud);
  //   }
  //   viewer->spinOnce(10);
  // }

  return 0;
}
