#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl/correspondence.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

namespace vllm_ros
{
void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id);

void publishPointcloud(ros::Publisher& publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// void publishWeights(ros::Publisher& publisher, const std::vector<float>& weights);

void publishImage(image_transport::Publisher& publisher, const cv::Mat& image);

void publishCorrespondences(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondences);

void publishTrajectory(ros::Publisher& publisher, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& trajectory, int color);

std::function<void(const sensor_msgs::ImageConstPtr&)> imageCallbackGenerator(cv::Mat& subscribed_image);

}  // namespace vllm_ros