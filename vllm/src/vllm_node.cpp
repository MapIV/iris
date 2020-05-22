#include "core/types.hpp"
#include "map/map.hpp"
#include "system/system.hpp"
#include <chrono>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/opencv.hpp>

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
Eigen::Matrix4f T_recover = Eigen::Matrix4f::Zero();
void callbackForRecover(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
  ROS_INFO("It subscribes initial_pose");

  float x = static_cast<float>(msg->pose.pose.position.x);
  float y = static_cast<float>(msg->pose.pose.position.y);
  float qw = static_cast<float>(msg->pose.pose.orientation.w);
  float qz = static_cast<float>(msg->pose.pose.orientation.z);

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
  // TODO:
  // We must set the values of the following parameters using rosparam
  // - string: config_path
  // - string: pcd_path

  vllm::Config config("src/vllm/config/hongo.yaml");

  return 0;
  ros::init(argc, argv, "vllm_node");
  ros::NodeHandle nh;

  // Setup for vslam_data
  vllm::pcXYZ::Ptr vslam_points(new vllm::pcXYZ);
  vllm::pcNormal::Ptr vslam_normals(new vllm::pcNormal);
  std::vector<float> vslam_weights;

  // Setup subscriber
  ros::Subscriber vslam_subscriber = nh.subscribe<pcl::PointCloud<pcl::PointXYZINormal>>("vllm/vslam_data", 5, callback);
  ros::Subscriber recover_pose_subscriber = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 5, callbackForRecover);
  tf::TransformListener listener;

  // Setup publisher
  ros::Publisher target_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("vllm/target_pointcloud", 1);
  ros::Publisher whole_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("vllm/whole_pointcloud", 1);
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("vllm/source_pointcloud", 1);
  ros::Publisher vllm_trajectory_publisher = nh.advertise<visualization_msgs::Marker>("vllm/vllm_trajectory", 1);
  ros::Publisher vslam_trajectory_publisher = nh.advertise<visualization_msgs::Marker>("vllm/vslam_trajectory", 1);
  ros::Publisher correspondences_publisher = nh.advertise<visualization_msgs::Marker>("vllm/correspondences", 1);
  vllm::Publication publication;


  // Initialize config
  // vllm::Config config("src/vllm/config/hongo.yaml");
  // vllm::Config config("hongo.yaml");
  // config.T_init.topLeftCorner(3, 3) = Eigen::AngleAxisf(-130.0 / 180.0 * 3.14, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  // std::cout << "T_init\n"
  //           << config.T_init << std::endl;

  // Load LiDAR map
  // vllm::map::Parameter map_param(
  //     "hongo.pcd", config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  // std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  // std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);
  std::chrono::system_clock::time_point m_start;

  ros::Rate loop_rate(10);
  int loop_count = 0;

  // Main loop
  while (ros::ok()) {
    ROS_INFO("LOOP");

    // Spin and wait
    ros::spinOnce();
    loop_rate.sleep();
  }

  ROS_INFO("Finalize the system");
  return 0;
}
