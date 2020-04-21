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
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

int main(int argc, char* argv[])
{
  // Initialzie ROS & subscriber
  ros::init(argc, argv, "vslam_node");

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
  // TODO: the topic name varis depending on user data.
  image_transport::Subscriber image_subscriber = it.subscribe("camera/color/image_raw", 1, vllm_ros::imageCallbackGenerator(subscribed_image));

  // Setup publisher
  ros::Publisher vslam_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("vllm/vslam_data", 1);
  image_transport::Publisher image_publisher = it.advertise("vllm/processed_image", 1);

  // Initialize config
  vllm::Config config(config_file_path->value());

  // Setup for OpenVSLAM
  vllm::BridgeOpenVSLAM bridge;
  bridge.setup(config);
  // output data
  vllm::pcXYZ::Ptr vslam_points(new vllm::pcXYZ);
  vllm::pcNormal::Ptr vslam_normals(new vllm::pcNormal);
  std::vector<float> vslam_weights;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr vslam_data(new pcl::PointCloud<pcl::PointXYZINormal>);

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

      // TODO: accuracy depends on alignment_node
      // Update threshold to adjust the number of points
      if (vslam_points->size() < 300 && accuracy > 0.10) accuracy -= 0.01f;
      if (vslam_points->size() > 500 && accuracy < 0.90) accuracy += 0.01f;

      vslam_data->clear();
      for (int i = 0; i < vslam_points->size(); i++) {
        pcl::PointXYZINormal p;
        p.x = vslam_points->at(i).x;
        p.y = vslam_points->at(i).y;
        p.z = vslam_points->at(i).z;
        p.normal_x = vslam_normals->at(i).normal_x;
        p.normal_y = vslam_normals->at(i).normal_y;
        p.normal_z = vslam_normals->at(i).normal_z;
        p.intensity = vslam_weights.at(i);
        vslam_data->push_back(p);
      }

      vllm_ros::publishImage(image_publisher, bridge.getFrame());
      {
        pcl_conversions::toPCL(ros::Time::now(), vslam_data->header.stamp);
        vslam_data->header.frame_id = "world";
        vslam_publisher.publish(vslam_data);
      }

      ROS_INFO("vslam update");
    }
    vllm_ros::publishPose(bridge.getCameraPose().inverse(), "vllm/vslam_pose");

    // Spin and wait
    ros::spinOnce();
    loop_10Hz.sleep();
  }

  ROS_INFO("Finalize the system");
  return 0;
}
