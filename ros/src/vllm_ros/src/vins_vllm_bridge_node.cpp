#include <chrono>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

// TODO: I don't like the function decleared in global scope like this
pcl::PointCloud<pcl::PointXYZ>::Ptr vins_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr vins_historycloud(new pcl::PointCloud<pcl::PointXYZ>);
bool vins_update = false;

void callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  ROS_INFO("It subscribes vslam_data %lu", msg->data.size());
  // *vins_pointcloud = *msg;
  // if (vins_pointcloud->size() > 0)
  //   vslam_update = true;
}

// TODO: I don't like the function decleared in global scope like this
Eigen::Matrix4f listenTransform(tf::TransformListener& listener)
{
  tf::StampedTransform transform;
  try {
    listener.lookupTransform("world", "vllm/vslam_pose", ros::Time(0), transform);
  } catch (...) {
  }

  double data[16];
  transform.getOpenGLMatrix(data);
  Eigen::Matrix4d T(data);
  return T.cast<float>();
}

int main(int argc, char* argv[])
{
  // Initialzie ROS & subscriber
  ros::init(argc, argv, "vins_vllm_bridge_node");

  ros::NodeHandle nh;

  // Setup subscriber
  ros::Subscriber pointcloud_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("vins_estimator/point_cloud", 5, callback);

  tf::TransformListener listener;

  // Setup publisher
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("vllm/vslam_data", 1);

  ros::Rate loop_rate(20);

  // Main loop
  while (ros::ok()) {

    Eigen::Matrix4f T_vslam = listenTransform(listener);
    ROS_INFO("%lu", vins_pointcloud->size());

    // Spin and wait
    ros::spinOnce();
    loop_rate.sleep();
  }

  ROS_INFO("Finalize the brdige");
  return 0;
}
