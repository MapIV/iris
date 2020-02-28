#pragma once
#include <Eigen/Dense>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

// L2 norm is used
pcl::CorrespondencesPtr getCorrespondences(const pcXYZ::Ptr& cloud_source, const pcXYZ::Ptr& cloud_target);

// get scale factor from rotation matrix
float getScale(const Eigen::Matrix3f& R);

// Not compatible with Scaling
Eigen::Matrix4f icpWithPurePCL(const pcXYZ::Ptr& cloud_query, const pcXYZ::Ptr& cloud_reference);

pcl::PointCloud<pcl::PointXYZ>::Ptr loadMapPointCloud(const std::string& pcd_file, float leaf = -1.0f);

pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& pcd_file);

pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(const pcXYZ::Ptr& cloud, float leaf);

void shufflePointCloud(pcXYZ::Ptr& cloud);

void wait(float ms);

Eigen::Matrix3f randomRotation();

Eigen::Matrix3f getNormalizedRotation(const Eigen::Matrix4f& T);

void transformNormals(
    const pcl::PointCloud<pcl::Normal>& source,
    pcl::PointCloud<pcl::Normal>& target,
    const Eigen::Matrix4f& T);

// pcl::PointCloud<pcl::Normal>::Ptr transformNormals(
//     const pcl::PointCloud<pcl::Normal>& source,
//     // pcl::PointCloud<pcl::Normal>& target,
//     const Eigen::Matrix4f& T);

}  // namespace vllm