#include "vllm/bridge/bridge.hpp"
#include "vllm/core/config.hpp"
#include "vllm_ros/communication.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

int main(int argc, char* argv[])
{
  // Initialzie ROS & subscriber
  ros::init(argc, argv, "vllm_node");

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
    std::cout << op.help() << std::endl;
    exit(EXIT_FAILURE);
  }

  // Setup subscriber
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  cv::Mat subscribed_image;
  image_transport::Subscriber image_subscriber = it.subscribe("camera/color/image_raw", 1, vllm_ros::imageCallbackGenerator(subscribed_image));

  // Setup publisher
  ros::Publisher points_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("vllm/source_points", 1);
  ros::Publisher normals_publisher = nh.advertise<pcl::PointCloud<pcl::Normal>>("vllm/source_normals", 1);
  image_transport::Publisher image_publisher = it.advertise("vllm/image", 1);

  // Initialize config
  vllm::Config config(config_file_path->value());

  // Setup for OpenVSLAM
  vllm::pcXYZ::Ptr vslam_points(new vllm::pcXYZ);
  vllm::pcNormal::Ptr vslam_normals(new vllm::pcNormal);
  std::vector<float> vslam_weights;
  vllm::BridgeOpenVSLAM bridge;
  bridge.setup(config);

  ros::Rate loop_10Hz(10);
  float accuracy = 0.5f;

  // Main loop
  while (ros::ok()) {
    if (!subscribed_image.empty()) {

      // process OpenVSLAM
      bridge.execute(subscribed_image);
      bridge.setCriteria(30, accuracy);
      bridge.getLandmarksAndNormals(vslam_points, vslam_normals, vslam_weights);
      subscribed_image = cv::Mat();  // Reset input

      // Update threshold to adjust the number of points
      if (vslam_points->size() < 300 && accuracy > 0.10) accuracy -= 0.01f;
      if (vslam_points->size() > 500 && accuracy < 0.90) accuracy += 0.01f;

      vllm_ros::publishImage(image_publisher, bridge.getFrame());
      // vllm_ros::publishPointcloud(source_pc_publisher, publication.cloud);
      // vllm_ros::publishPose(publication.offset_camera, "vslam_pose");
      ROS_INFO("kusa");
    }

    // Spin and wait
    ros::spinOnce();
    loop_10Hz.sleep();
  }

  ROS_INFO("Finalize the system");
  return 0;
}
