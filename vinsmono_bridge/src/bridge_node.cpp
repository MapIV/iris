#include <chrono>
#include <list>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
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
pcl::PointCloud<pcl::PointXYZINormal>::Ptr pushbackPointXYZINormal(const sensor_msgs::PointCloudConstPtr& msg, const Eigen::Vector3f& camera_pos);

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vinsmono_bridge_node");
  ros::NodeHandle nh;

  // Setup subscriber
  tf::TransformListener listener;
  message_filters::Subscriber<sensor_msgs::PointCloud> pointcloud_subscriber(nh, "vins_estimator/point_cloud", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud> historycloud_subscriber(nh, "vins_estimator/history_cloud", 1);
  message_filters::TimeSynchronizer<sensor_msgs::PointCloud, sensor_msgs::PointCloud> sync(pointcloud_subscriber, historycloud_subscriber, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  // Setup publisher
  tf::TransformBroadcaster tf_broadcaster;
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("iris/vslam_data", 1);
  ros::Rate loop_rate(20);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr vins_pointcloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  std::list<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> pointcloud_history;

  // Conversion of the camera's direction of travel to the z-axis
  Eigen::Matrix4f T_align;
  T_align << 1, 0, 0, 0,
      0, 0, -1, 0,
      0, 1, 0, 0,
      0, 0, 0, 1;

  // Start main loop
  ROS_INFO("start main loop.");
  while (ros::ok()) {
    Eigen::Matrix4f T = T_align * listenTransform(listener);

    if (vins_update) {
      vins_update = false;

      const Eigen::Vector3f camera_pos = T.topRightCorner(3, 1);
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr active_cloud = pushbackPointXYZINormal(tmp_msg1, camera_pos);
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr inactive_cloud = pushbackPointXYZINormal(tmp_msg2, camera_pos);
      pointcloud_history.push_front(inactive_cloud);

      if (pointcloud_history.size() > 300)
        pointcloud_history.pop_back();


      vins_pointcloud->clear();
      *vins_pointcloud += *active_cloud;
      for (auto itr = pointcloud_history.begin(); itr != pointcloud_history.end();) {
        *vins_pointcloud += **itr;
        for (int i = 0; i < 10; i++) {
          if (itr != pointcloud_history.end())
            itr++;
        }
        if (vins_pointcloud->size() > 500) break;
      }
      std::cout << "vins_cloud size= " << vins_pointcloud->size() << ", active_cloud size= " << active_cloud->size() << std::endl;

      // Publish
      pcl_conversions::toPCL(ros::Time::now(), vins_pointcloud->header.stamp);
      vins_pointcloud->header.frame_id = "world";
      source_pc_publisher.publish(vins_pointcloud);
    }

    // Publish TF
    tf::StampedTransform transform;
    transform.setFromOpenGLMatrix(T.cast<double>().eval().data());
    tf_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "iris/vslam_pose"));

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

pcl::PointCloud<pcl::PointXYZINormal>::Ptr pushbackPointXYZINormal(const sensor_msgs::PointCloudConstPtr& msg, const Eigen::Vector3f& camera_pos)
{
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);

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

  // Conversion of the camera's direction of travel to the z-axis
  Eigen::Matrix4f T_align;
  T_align << 1, 0, 0, 0,
      0, 0, -1, 0,
      0, 1, 0, 0,
      0, 0, 0, 1;

  pcl::transformPointCloud(*cloud, *cloud, T_align);

  return cloud;
}
