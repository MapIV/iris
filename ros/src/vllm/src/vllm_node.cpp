#include "core/config.hpp"
#include "map/map.hpp"
#include "system/system.hpp"
#include "viewer/pangolin_viewer.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <ros/ros.h>

cv::Mat subscribed_image;
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  subscribed_image = cv_ptr->image.clone();
}

int main(int argc, char* argv[])
{
  // Initialzie ROS & subscriber
  ros::init(argc, argv, "vllm_node");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("camera/color/image_raw", 1, &imageCallback);

  // Analyze arugments
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


  // Initialize config
  vllm::Config config(config_file_path->value());

  // Load LiDAR map
  vllm::map::Parameter map_param(
      config.pcd_file, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);
  std::chrono::system_clock::time_point m_start;

  // Initialize viewer
  vllm::viewer::PangolinViewer pangolin_viewer(system);
  pangolin_viewer.startLoop();
  cv::namedWindow("VLLM", cv::WINDOW_AUTOSIZE);

  // The main loop runs at this frequency at most
  const float hz = 10;
  ros::Rate loop_rate(hz);

  // Main loop
  while (ros::ok()) {
    m_start = std::chrono::system_clock::now();

    if (!subscribed_image.empty()) {
      // Execution
      system->execute(subscribed_image);

      // visualize by OpenCV
      cv::imshow("VLLM", system->getFrame());
      if (cv::waitKey(1) == 'q') break;
    }

    std::cout << "time= \033[35m"
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
              << "\033[m ms" << std::endl;

    // spin & wait
    ros::spinOnce();
    loop_rate.sleep();
  }

  // Stop viewer
  pangolin_viewer.quitLoop();
  return 0;
}