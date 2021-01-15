// Copyright (c) 2020, Map IV, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the Map IV, Inc. nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "publish/publish.hpp"
#include "core/util.hpp"
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace iris
{
// h: [0,360]
// s: [0,1]
// v: [0,1]
Eigen::Vector3f convertRGB(Eigen::Vector3f hsv)
{
  const float max = hsv(2);
  const float min = max * (1 - hsv(1));
  const float H = hsv(0);
  const float D = max - min;
  if (H < 60) return {max, H / 60.f * D + min, min};
  if (H < 120) return {(120.f - H) / 60.f * D + min, max, min};
  if (H < 180) return {min, max, (H - 120) / 60.f * D + min};
  if (H < 240) return {min, (240.f - H) / 60.f * D + min, max};
  if (H < 300) return {(H - 240.f) / 60.f * D + min, min, max};
  if (H < 360) return {max, min, (360.f - H) / 60.f * D + min};
  return {1.0f, 1.0f, 1.0f};
}

void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id)
{
  static tf::TransformBroadcaster br;


  tf::Transform transform;
  transform.setFromOpenGLMatrix(util::normalizePose(T).cast<double>().eval().data());
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

void publishCorrespondences(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondences)
{
  visualization_msgs::Marker line_strip;
  line_strip.header.frame_id = "world";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "correspondences";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w = 1.0;
  line_strip.id = 0;
  line_strip.scale.x = 0.15;
  line_strip.type = visualization_msgs::Marker::LINE_LIST;
  line_strip.color.r = 1.0;
  line_strip.color.g = 0.0;
  line_strip.color.b = 0.0;
  line_strip.color.a = 1.0;

  for (const pcl::Correspondence& c : *correspondences) {
    pcl::PointXYZ point1 = source->at(c.index_query);
    pcl::PointXYZ point2 = target->at(c.index_match);
    geometry_msgs::Point p1, p2;
    p1.x = point1.x;
    p1.y = point1.y;
    p1.z = point1.z;
    p2.x = point2.x;
    p2.y = point2.y;
    p2.z = point2.z;
    line_strip.points.push_back(p1);
    line_strip.points.push_back(p2);
  }

  publisher.publish(line_strip);
}

void publishNormal(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
  visualization_msgs::MarkerArray marker_array;

  geometry_msgs::Vector3 arrow;
  arrow.x = 0.2;  // length
  arrow.y = 0.4;  // width
  arrow.z = 0.5;  // height

  for (size_t id = 0; id < cloud->size(); id++) {
    visualization_msgs::Marker marker;

    pcl::PointXYZ p = cloud->at(id);
    pcl::Normal n = normals->at(id);

    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "normal";
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = static_cast<int>(id);
    marker.scale = arrow;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    geometry_msgs::Point p1, p2;
    p1.x = p.x;
    p1.y = p.y;
    p1.z = p.z;
    p2.x = p.x + 2.0f * n.normal_x;
    p2.y = p.y + 2.0f * n.normal_y;
    p2.z = p.z + 2.0f * n.normal_z;

    marker.points.push_back(p1);
    marker.points.push_back(p2);
    marker_array.markers.push_back(marker);
  }
  publisher.publish(marker_array);
}

Eigen::Quaternionf normalToQuaternion(const Eigen::Vector3f& n)
{
  return Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), n);
}

void publishCovariance(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
  visualization_msgs::MarkerArray marker_array;

  geometry_msgs::Vector3 diameter;
  diameter.x = 4.0;
  diameter.y = 4.0;
  diameter.z = 1.0;

  for (size_t id = 0; id < cloud->size(); id++) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "covariance";
    marker.action = visualization_msgs::Marker::ADD;

    pcl::PointXYZ p = cloud->at(id);
    marker.pose.position.x = p.x;
    marker.pose.position.y = p.y;
    marker.pose.position.z = p.z;

    pcl::Normal n = normals->at(id);
    Eigen::Quaternionf q = normalToQuaternion(Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z));
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    marker.id = static_cast<int>(id);
    marker.scale = diameter;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.3f;
    marker_array.markers.push_back(marker);
  }

  publisher.publish(marker_array);
}

visualization_msgs::Marker makeMarkerAsLine(const Eigen::Vector3f& s, const Eigen::Vector3f& e, int id)
{
  visualization_msgs::Marker line_strip;
  line_strip.header.frame_id = "world";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "points_and_lines";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w = 1.0;
  line_strip.id = id;
  line_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line_strip.color.a = 1.0;
  line_strip.scale.x = 0.5;

  {
    geometry_msgs::Point p;
    p.x = s.x();
    p.y = s.y();
    p.z = s.z();
    line_strip.points.push_back(p);
  }
  {
    geometry_msgs::Point p;
    p.x = e.x();
    p.y = e.y();
    p.z = e.z();
    line_strip.points.push_back(p);
  }
  return line_strip;
}

void publishPath(ros::Publisher& publisher, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& trajectory)
{
  nav_msgs::Path path;

  path.header.frame_id = "world";
  path.header.stamp = ros::Time::now();
  for (const Eigen::Vector3f& t : trajectory) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = t.x();
    pose_stamped.pose.position.y = t.y();
    pose_stamped.pose.position.z = t.z();

    // We don't care about the orientation
    pose_stamped.pose.orientation.w = 1;
    path.poses.push_back(pose_stamped);
  }
  publisher.publish(path);
}

std::function<void(const sensor_msgs::ImageConstPtr&)> imageCallbackGenerator(cv::Mat& subscribed_image)
{
  return [&subscribed_image](const sensor_msgs::ImageConstPtr& msg) -> void {
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    subscribed_image = cv_ptr->image.clone();
  };
}

std::function<void(const sensor_msgs::CompressedImageConstPtr&)> compressedImageCallbackGenerator(cv::Mat& subscribed_image)
{
  return [&subscribed_image](const sensor_msgs::CompressedImageConstPtr& msg) -> void {
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    subscribed_image = cv_ptr->image.clone();
  };
}


void publishResetPointcloud(ros::Publisher& publisher)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
  pcl_conversions::toPCL(ros::Time::now(), tmp->header.stamp);
  tmp->header.frame_id = "world";
  publisher.publish(tmp);
}

void publishResetCorrespondences(ros::Publisher& publisher)
{
  visualization_msgs::Marker line_strip;
  line_strip.header.frame_id = "world";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "correspondences";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w = 1.0;
  line_strip.id = 0;
  line_strip.scale.x = 0.3;
  line_strip.type = visualization_msgs::Marker::LINE_LIST;
  line_strip.color.r = 1.0;
  line_strip.color.g = 0.0;
  line_strip.color.b = 0.0;
  line_strip.color.a = 1.0;
  publisher.publish(line_strip);
}


}  // namespace iris
