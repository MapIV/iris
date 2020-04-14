#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

namespace vllm
{
void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id)
{
  static tf::TransformBroadcaster br;

  geometry_msgs::Pose t_pose;
  t_pose.position.x = T(0, 3);
  t_pose.position.y = T(1, 3);
  t_pose.position.z = T(2, 3);

  Eigen::Matrix3f R = T.topLeftCorner(3, 3);
  Eigen::Quaternionf q(R);
  t_pose.orientation.w = q.w();
  t_pose.orientation.x = q.x();
  t_pose.orientation.y = q.y();
  t_pose.orientation.z = q.z();

  tf::Transform transform;
  poseMsgToTF(t_pose, transform);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", child_frame_id));
}

void publishPointcloud(ros::Publisher& publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  auto tmp = cloud;
  pcl_conversions::toPCL(ros::Time::now(), tmp->header.stamp);
  tmp->header.frame_id = "world";
  publisher.publish(tmp);
}

void publishImage(image_transport::Publisher& publisher, const cv::Mat& image)
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
  publisher.publish(msg);
}

void publishTrajectory(ros::Publisher& publisher,
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& trajectory)
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

  for (const Eigen::Vector3f& t : trajectory) {
    geometry_msgs::Point p;
    p.x = t.x();
    p.y = t.y();
    p.z = t.z();
    line_strip.points.push_back(p);
  }
  publisher.publish(line_strip);
}

std::function<void(const sensor_msgs::ImageConstPtr&)> imageCallbackGenerator(cv::Mat& subscribed_image)
{
  return [&subscribed_image](const sensor_msgs::ImageConstPtr& msg) -> void {
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    subscribed_image = cv_ptr->image.clone();
  };
}

}  // namespace vllm