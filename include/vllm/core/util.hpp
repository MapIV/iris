#pragma once
#include "vllm/core/types.hpp"
#include <Eigen/Dense>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
namespace util
{
float getScale(const Eigen::MatrixXf& A);
Eigen::Matrix4f applyScaling(const Eigen::Matrix4f& T, float scale);
Eigen::Matrix3f normalizeRotation(const Eigen::MatrixXf& A);
Eigen::Matrix4f normalizePose(const Eigen::Matrix4f& T);

Eigen::Matrix4f make3DPoseFrom2DPose(float x, float y, float nx, float ny);

// load
// pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& pcd_file);
// pcl::PointCloud<pcl::PointXYZ>::Ptr loadMapPointCloud(const std::string& pcd_file, float leaf = -1.0f);
void loadMap(
    const std::string& pcd_file,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    float grid_leaf, float radius);

// normal
pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(const pcXYZ::Ptr& cloud, float leaf);
void transformNormals(const pcNormal& source, pcNormal& target, const Eigen::Matrix4f& T);

// randomize
Eigen::Matrix3f randomRotation();
void shufflePointCloud(pcXYZ::Ptr& cloud);

}  // namespace util
}  // namespace vllm