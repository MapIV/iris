#include "vllm/bridge/bridge.hpp"
#include "vllm/core/config.hpp"
#include "vllm/map/map.hpp"
#include "vllm/system/system.hpp"
#include "vllm/viewer/pangolin_viewer.hpp"
#include "vllm_ros/communication.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// TODO: I don't like the function decleared in global scope like this
pcl::PointCloud<pcl::PointXYZINormal>::Ptr vslam_data(new pcl::PointCloud<pcl::PointXYZINormal>);
bool vslam_update = false;
void callback(const pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr& msg)
{
  ROS_INFO("It subscribes vslam_data %lu", msg->size());
  *vslam_data = *msg;
  if (vslam_data->size() > 0)
    vslam_update = true;
}

// TODO: I don't like the function decleared in global scope like this
Eigen::Matrix4f listenTransform(tf::TransformListener& listener)
{
  tf::StampedTransform transform;
  try {
    listener.waitForTransform("world", "vllm/vslam_pose", ros::Time(0), ros::Duration(10.0));
    listener.lookupTransform("world", "vllm/vslam_pose", ros::Time(0), transform);
  } catch (...) {
  }

  double data[16];
  transform.getOpenGLMatrix(data);
  Eigen::Matrix4d T(data);
  return T.cast<float>();
}

// TODO: I don't like the function decleared in global scope like this
Eigen::Matrix4f T_recover = Eigen::Matrix4f::Zero();
void callbackForRecover(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
  ROS_INFO("It subscribes initial_pose");

  float x = msg->pose.pose.position.x;
  float y = msg->pose.pose.position.y;
  float qw = msg->pose.pose.orientation.w;
  float qz = msg->pose.pose.orientation.z;

  T_recover.setIdentity();
  T_recover(0, 3) = x;
  T_recover(1, 3) = y;
  float theta = 2 * std::atan2(qz, qw);
  Eigen::Matrix3f R;
  R << 0, 0, 1,
      -1, 0, 0,
      0, -1, 0;
  T_recover.topLeftCorner(3, 3) = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()).toRotationMatrix() * R;
}

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

  ros::NodeHandle nh;

  // Setup for OpenVSLAM
  vllm::pcXYZ::Ptr vslam_points(new vllm::pcXYZ);
  vllm::pcNormal::Ptr vslam_normals(new vllm::pcNormal);
  std::vector<float> vslam_weights;

  // Setup subscriber
  ros::Subscriber vslam_subscriber = nh.subscribe<pcl::PointCloud<pcl::PointXYZINormal>>("vllm/vslam_data", 1, callback);
  ros::Subscriber recover_pose_subscriber = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, callbackForRecover);
  tf::TransformListener listener;

  // Setup publisher
  ros::Publisher target_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("vllm/target_pointcloud", 1);
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("vllm/source_pointcloud", 1);
  ros::Publisher vllm_trajectory_publisher = nh.advertise<visualization_msgs::MarkerArray>("vllm/vllm_trajectory", 1);
  ros::Publisher vslam_trajectory_publisher = nh.advertise<visualization_msgs::MarkerArray>("vllm/vslam_trajectory", 1);
  ros::Publisher correspondences_publisher = nh.advertise<visualization_msgs::Marker>("vllm/correspondences", 1);
  vllm::Publication publication;

  // Initialize config
  vllm::Config config(config_file_path->value());

  // Load LiDAR map
  vllm::map::Parameter map_param(
      config.pcd_file, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);
  std::chrono::system_clock::time_point m_start;

  // Publish for rviz to clear previous visualization
  {
    vllm_ros::publishResetPointcloud(source_pc_publisher);
    vllm_ros::publishResetTrajectory(vllm_trajectory_publisher);
    vllm_ros::publishResetTrajectory(vslam_trajectory_publisher);
    vllm_ros::publishResetCorrespondences(correspondences_publisher);
    vllm_ros::publishPose(Eigen::Matrix4f::Zero(), "vllm/offseted_vslam_pose");
    vllm_ros::publishPose(Eigen::Matrix4f::Zero(), "vllm/vllm_pose");
  }

  ros::Rate loop_rate(20);
  int loop_count = 0;

  // Main loop
  while (ros::ok()) {

    Eigen::Matrix4f T_vslam = listenTransform(listener);

    if (!T_recover.isZero()) {
      system->specifyTWorld(T_recover);
      T_recover.setZero();
    }

    if (vslam_update) {
      vslam_update = false;
      m_start = std::chrono::system_clock::now();

      // Execution
      system->execute(2, T_vslam, vslam_data);

      // Publish for rviz
      system->popPublication(publication);
      vllm_ros::publishPointcloud(source_pc_publisher, publication.cloud);
      vllm_ros::publishTrajectory(vllm_trajectory_publisher, publication.vllm_trajectory, 0);
      vllm_ros::publishTrajectory(vslam_trajectory_publisher, publication.offset_trajectory, 1);
      vllm_ros::publishCorrespondences(correspondences_publisher, publication.cloud, map->getTargetCloud(), publication.correspondences);
      vllm_ros::publishPose(publication.offset_camera, "vllm/offseted_vslam_pose");
      vllm_ros::publishPose(publication.vllm_camera, "vllm/vllm_pose");

      // Inform processing time
      std::stringstream ss;
      ss << "processing time= \033[35m"
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
         << "\033[m ms";
      ROS_INFO("VLLM/ALIGN: %s", ss.str().c_str());

    } else {
    }

    // Publish target pointcloud map
    if (++loop_count >= 10) {
      loop_count = 0;
      vllm_ros::publishPointcloud(target_pc_publisher, map->getTargetCloud());
    }

    // Spin and wait
    ros::spinOnce();
    loop_rate.sleep();
  }

  ROS_INFO("Finalize the system");
  return 0;
}