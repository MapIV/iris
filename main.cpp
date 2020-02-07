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
  BridgeOpenVSLAM bridge;
  std::thread vslam_thread = std::thread(&BridgeOpenVSLAM::start, &bridge, argc, argv);

  std::cout << "====== try get frame publisher" << std::endl;
  const auto frame_publisher = bridge.get_frame_publisher();
  const auto map_publisher = bridge.get_map_publisher();
  std::cout << "====== got frame publisher" << std::endl;

  pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("PCLVisualizer"));
  viewer->addCoordinateSystem(0.5);

  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  while (!viewer->wasStopped()) {
    cv::imshow("OpenCV", frame_publisher->draw_frame());
    cv::waitKey(10);

    std::vector<openvslam::data::landmark*> landmarks;
    std::set<openvslam::data::landmark*> local_landmarks;
    map_publisher->get_landmarks(landmarks, local_landmarks);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = showLandmarks(landmarks, local_landmarks);
    if (cloud->size() > 0) {
      viewer->removeAllPointClouds();
      viewer->addPointCloud(cloud);
    }
    viewer->spinOnce(10);
  }


  vslam_thread.join();
  return 0;
}
