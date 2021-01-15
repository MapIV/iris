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

#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl/correspondence.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>

namespace iris
{
void publishImage(image_transport::Publisher& publisher, const cv::Mat& image);

void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id);
void publishPointcloud(ros::Publisher& publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
void publishPath(ros::Publisher& publisher, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& path);
void publishCorrespondences(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondences);
void publishNormal(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals);
void publishCovariance(ros::Publisher& publisher,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals);

void publishResetPointcloud(ros::Publisher& publisher);
void publishResetCorrespondences(ros::Publisher& publisher);

std::function<void(const sensor_msgs::ImageConstPtr&)> imageCallbackGenerator(cv::Mat& subscribed_image);
std::function<void(const sensor_msgs::CompressedImageConstPtr&)> compressedImageCallbackGenerator(cv::Mat& subscribed_image);

}  // namespace iris
