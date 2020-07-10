#include "core/types.hpp"
#include "map/map.hpp"
#include "publish/publish.hpp"
#include "system/system.hpp"
#include <chrono>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
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
    listener.lookupTransform("world", "iris/vslam_pose", ros::Time(0), transform);
  } catch (...) {
  }

  double data[16];
  transform.getOpenGLMatrix(data);
  Eigen::Matrix4d T(data);
  return T.cast<float>();
}

// TODO: I don't like the function decleared in global scope like this
Eigen::Matrix4f T_recover = Eigen::Matrix4f::Zero();
pcl::PointCloud<pcl::PointXYZ>::Ptr current_pointcloud = nullptr;
void callbackForRecover(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
  ROS_INFO("It subscribes initial_pose");

  float x = static_cast<float>(msg->pose.pose.position.x);
  float y = static_cast<float>(msg->pose.pose.position.y);
  float qw = static_cast<float>(msg->pose.pose.orientation.w);
  float qz = static_cast<float>(msg->pose.pose.orientation.z);

  float z = std::numeric_limits<float>::max();

  if (current_pointcloud != nullptr) {
    z = 0;
  } else {
    for (const pcl::PointXYZ& p : *current_pointcloud) {
      constexpr float r2 = 2 * 2;  // [m^2]
      float dx = x - p.x;
      float dy = y - p.y;
      if (dx * dx + dy * dy < r2)
        z = std::min(z, p.z);
    }
  }

  T_recover.setIdentity();
  T_recover(0, 3) = x;
  T_recover(1, 3) = y;
  T_recover(2, 3) = z;
  float theta = 2 * std::atan2(qz, qw);
  Eigen::Matrix3f R;
  R << 0, 0, 1,
      -1, 0, 0,
      0, -1, 0;
  T_recover.topLeftCorner(3, 3) = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()).toRotationMatrix() * R;
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "iris_node");
  ros::NodeHandle nh;

  // Setup for vslam_data
  iris::pcXYZ::Ptr vslam_points(new iris::pcXYZ);
  iris::pcNormal::Ptr vslam_normals(new iris::pcNormal);
  std::vector<float> vslam_weights;

  // Setup subscriber
  ros::Subscriber vslam_subscriber = nh.subscribe<pcl::PointCloud<pcl::PointXYZINormal>>("iris/vslam_data", 5, callback);
  ros::Subscriber recover_pose_subscriber = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 5, callbackForRecover);
  tf::TransformListener listener;

  // Setup publisher
  ros::Publisher target_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("iris/target_pointcloud", 1);
  ros::Publisher whole_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("iris/whole_pointcloud", 1);
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("iris/source_pointcloud", 1);
  ros::Publisher iris_trajectory_publisher = nh.advertise<visualization_msgs::Marker>("iris/iris_trajectory", 1);
  ros::Publisher vslam_trajectory_publisher = nh.advertise<visualization_msgs::Marker>("iris/vslam_trajectory", 1);
  ros::Publisher correspondences_publisher = nh.advertise<visualization_msgs::Marker>("iris/correspondences", 1);
  ros::Publisher scale_publisher = nh.advertise<std_msgs::Float32>("iris/align_scale", 1);
  iris::Publication publication;

  // Get rosparams
  ros::NodeHandle pnh("~");
  std::string config_path, pcd_path;
  pnh.getParam("iris_config_path", config_path);
  pnh.getParam("pcd_path", pcd_path);
  ROS_INFO("config_path: %s, pcd_path: %s", config_path.c_str(), pcd_path.c_str());

  // Initialize config
  iris::Config config(config_path);

  // Load LiDAR map
  iris::map::Parameter map_param(
      pcd_path, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<iris::map::Map> map = std::make_shared<iris::map::Map>(map_param, config.T_init);

  // Initialize system
  std::shared_ptr<iris::System> system = std::make_shared<iris::System>(config, map);

  std::chrono::system_clock::time_point m_start;
  ros::Rate loop_rate(10);
  int loop_count = 0;

  Eigen::Matrix4f offseted_vslam_pose = config.T_init;
  Eigen::Matrix4f iris_pose = config.T_init;

  // Start main loop
  ROS_INFO("start main loop.");
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
      iris::publishPointcloud(source_pc_publisher, publication.cloud);
      iris::publishTrajectory(iris_trajectory_publisher, publication.iris_trajectory, {1.0f, 0.0f, 1.0f});
      iris::publishTrajectory(vslam_trajectory_publisher, publication.offset_trajectory, {0.6f, 0.6f, 0.6f});
      iris::publishCorrespondences(correspondences_publisher, publication.cloud, map->getTargetCloud(), publication.correspondences);
      {
        std_msgs::Float32 scale;
        scale.data = iris::util::getScale(publication.T_align);
        scale_publisher.publish(scale);
      }
      offseted_vslam_pose = publication.offset_camera;
      iris_pose = publication.iris_camera;

      current_pointcloud = publication.cloud;

      // Inform processing time
      std::stringstream ss;
      ss << "processing time= \033[35m"
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
         << "\033[m ms";
      ROS_INFO("Iris/ALIGN: %s", ss.str().c_str());
    }

    iris::publishPose(offseted_vslam_pose, "iris/offseted_vslam_pose");
    iris::publishPose(iris_pose, "iris/iris_pose");

    // Publish target pointcloud map at long intervals
    if (++loop_count >= 10) {
      loop_count = 0;
      iris::publishPointcloud(target_pc_publisher, map->getTargetCloud());
      iris::publishPointcloud(whole_pc_publisher, map->getSparseCloud());
    }

    // Spin and wait
    ros::spinOnce();
    loop_rate.sleep();
  }

  ROS_INFO("Finalize the system");
  return 0;
}
