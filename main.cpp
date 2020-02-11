#include "bridge.hpp"
#include "openvslam/data/landmark.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include "pangolin_viewer.hpp"
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr convertLandmarks(
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
    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));

    cloud->push_back(p);
  }
  for (const auto local_lm : local_landmarks) {
    if (local_lm->will_be_erased()) {
      continue;
    }
    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));
    cloud->push_back(p);
  }

  return cloud;
}

int main(int argc, char* argv[])
{
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

    std::vector<openvslam::data::landmark*> landmarks;
    std::set<openvslam::data::landmark*> local_landmarks;
    map_publisher->get_landmarks(landmarks, local_landmarks);
    auto cloud = convertLandmarks(landmarks, local_landmarks);

    int state = static_cast<int>(frame_publisher->get_tracking_state());
    Eigen::Matrix4d camera = map_publisher->get_current_cam_pose();


    // Visualize by OpenCV
    cv::imshow("OpenCV", frame_publisher->draw_frame());

    // Visualize by Pangolin
    pangolin_viewer.clear();
    pangolin_viewer.drawGridLine();
    pangolin_viewer.drawState(state);
    pangolin_viewer.addPointCloud(cloud, {1.0f, 1.0f, 0.0f, 2.0f});
    pangolin_viewer.addPointCloud(cloud_map, {0.5f, 0.5f, 0.5f, 1.0f});
    pangolin_viewer.addCamera(camera, {0.0f, 1.0f, 0.0f, 2.0f});
    pangolin_viewer.swap();

    int key = cv::waitKey(10);
    if (key == 'q') {
      break;
    }
  }

  return 0;
}