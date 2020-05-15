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

bool vins_update = false;
sensor_msgs::PointCloudConstPtr tmp_msg1, tmp_msg2;

void callback(const sensor_msgs::PointCloudConstPtr& pointcloud_msg, const sensor_msgs::PointCloudConstPtr& historycloud_msg)
{
  ROS_INFO("visible: %lu, unvisible:%lu ", pointcloud_msg->points.size(), historycloud_msg->points.size());
  if (pointcloud_msg->points.empty() && historycloud_msg->points.empty()) return;
  vins_update = true;
  tmp_msg1 = pointcloud_msg;
  tmp_msg2 = historycloud_msg;
}

Eigen::Matrix4f listenTransform(tf::TransformListener& listener);
void pushbackPointXYZINormal(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const sensor_msgs::PointCloudConstPtr& msg, const Eigen::Vector3f& camera_pos);

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vins_vllm_bridge_node");

  ros::NodeHandle nh;

  // Setup subscriber
  tf::TransformListener listener;
  message_filters::Subscriber<sensor_msgs::PointCloud> pointcloud_subscriber(nh, "vins_estimator/point_cloud", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud> historycloud_subscriber(nh, "vins_estimator/history_cloud", 1);
  message_filters::TimeSynchronizer<sensor_msgs::PointCloud, sensor_msgs::PointCloud> sync(pointcloud_subscriber, historycloud_subscriber, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  // Setup publisher
  tf::TransformBroadcaster tf_broadcaster;
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("vllm/vslam_data", 1);
  ros::Rate loop_rate(20);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr vins_pointcloud(new pcl::PointCloud<pcl::PointXYZINormal>);

  // Main loop
  while (ros::ok()) {
    Eigen::Matrix4f T = listenTransform(listener);
    vins_pointcloud->clear();

    if (vins_update) {
      std::cout << T << std::endl;
      vins_update = false;

      const Eigen::Vector3f camera_pos = T.topRightCorner(3, 1);
      pushbackPointXYZINormal(vins_pointcloud, tmp_msg1, camera_pos);
      pushbackPointXYZINormal(vins_pointcloud, tmp_msg2, camera_pos);

      // Publish
      pcl_conversions::toPCL(ros::Time::now(), vins_pointcloud->header.stamp);
      vins_pointcloud->header.frame_id = "world";
      source_pc_publisher.publish(vins_pointcloud);
    }


    // Publish TF
    tf::StampedTransform transform;
    transform.setFromOpenGLMatrix(T.cast<double>().eval().data());
    tf_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "vllm/vslam_pose"));


    // Spin and wait
    ros::spinOnce();
    loop_rate.sleep();
  }

  ROS_INFO("Finalize the brdige");
  return 0;
}

Eigen::Matrix4f listenTransform(tf::TransformListener& listener)
{
  tf::StampedTransform transform;
  try {
    listener.lookupTransform("world", "body", ros::Time(0), transform);
  } catch (...) {
    return Eigen::Matrix4f::Identity();
  }
  double data[16];
  transform.getOpenGLMatrix(data);
  Eigen::Matrix4d T(data);
  return T.cast<float>();
}

void pushbackPointXYZINormal(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const sensor_msgs::PointCloudConstPtr& msg, const Eigen::Vector3f& camera_pos)
{
  for (size_t i = 0; i < msg->points.size(); i++) {
    const geometry_msgs::Point32& g_p = msg->points.at(i);
    pcl::PointXYZINormal point;
    point.x = g_p.x;
    point.y = g_p.y;
    point.z = g_p.z;
    point.intensity = 1.0f;

    Eigen::Vector3f normal;
    normal << g_p.x, g_p.y, g_p.z;
    normal = (normal - camera_pos).normalized();

    point.normal_x = normal.x();
    point.normal_y = normal.y();
    point.normal_z = normal.z();
    cloud->push_back(point);
  }
}