#include "vllm/core/config.hpp"
#include "vllm/map/map.hpp"
#include "vllm/system/system.hpp"
#include "vllm/viewer/pangolin_viewer.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>
#include <popl.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

cv::Mat subscribed_image;
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  subscribed_image = cv_ptr->image.clone();
}

const std::string WINDOW_NAME = "Visual Localization in 3D LiDAR Map";

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
    std::cout << op.help() << std::endl;
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
  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // The main loop runs at this frequency at most
  ros::Rate loop_10Hz(10);

  // Setup publisher
  ros::Publisher target_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("target_pointcloud", 10);
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("source_pointcloud", 10);
  ros::Publisher marker_publisher = nh.advertise<visualization_msgs::Marker>("visualization_marker", 0);
  image_transport::ImageTransport it(nh);
  image_transport::Publisher image_publisher = it.advertise("image", 10);

  // Main loop
  int loop_count = 0;
  while (ros::ok()) {
    m_start = std::chrono::system_clock::now();

    if (!subscribed_image.empty()) {
      // Execution
      system->execute(subscribed_image);
      subscribed_image = cv::Mat();  // reset input

      // NOTE: ## This is deep darkness. ##
      vllm::Publication p;
      bool flag = system->popPublication(p);

      std::cout << "FlAG:  " << std::boolalpha << flag << std::endl;

      // Publish image
      {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", system->getFrame()).toImageMsg();
        image_publisher.publish(msg);
      }

      // Publish source cloud
      {
        auto msg = system->ros_pointcloud;
        msg->header.frame_id = "world";
        pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
        source_pc_publisher.publish(msg);
      }

      // Publish trajectory
      {
        visualization_msgs::Marker line_strip;
        line_strip.header.frame_id = "world";
        line_strip.header.stamp = ros::Time::now();
        line_strip.ns = "points_and_lines";
        line_strip.action = visualization_msgs::Marker::ADD;
        line_strip.pose.orientation.w = 1.0;
        line_strip.id = 0;
        line_strip.scale.x = 0.1;
        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_strip.color.r = 0.0;
        line_strip.color.g = 1.0;
        line_strip.color.b = 0.0;
        line_strip.color.a = 1.0;

        for (const Eigen::Vector3f& t : system->vllm_trajectory) {
          geometry_msgs::Point p;
          p.x = t.x();
          p.y = t.y();
          p.z = t.z();
          line_strip.points.push_back(p);
        }
        marker_publisher.publish(line_strip);
      }

      // Publish vslam pose
      {
        Eigen::Matrix4f pose = system->ros_vslam_pose;
        geometry_msgs::Pose t_pose;
        t_pose.position.x = pose(0, 3);
        t_pose.position.y = pose(1, 3);
        t_pose.position.z = pose(2, 3);
        t_pose.orientation.w = 1.0;

        std::cout << "\nvslam_camera\n"
                  << pose << std::endl;

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        poseMsgToTF(t_pose, transform);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "vslam_pose"));
      }

      // Publish vllm pose
      {
        Eigen::Matrix4f pose = system->ros_vllm_pose;
        geometry_msgs::Pose t_pose;
        t_pose.position.x = pose(0, 3);
        t_pose.position.y = pose(1, 3);
        t_pose.position.z = pose(2, 3);
        t_pose.orientation.w = 1.0;

        std::cout << "\n vllm_camera\n"
                  << pose << std::endl;

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        poseMsgToTF(t_pose, transform);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "vllm_pose"));
      }
    }

    // inform processing time
    {
      std::stringstream ss;
      ss << "time= \033[35m"
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
         << "\033[m ms";
      ROS_INFO("%s", ss.str().c_str());
    }

    // publish heavy data
    if (++loop_count >= 100) {
      loop_count = 0;
      auto msg = map->getTargetCloud();
      msg->header.frame_id = "world";
      pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
      target_pc_publisher.publish(msg);
    }


    // spin & wait
    ros::spinOnce();
    loop_10Hz.sleep();
  }
  ROS_INFO("Finalize the system");

  return 0;
}