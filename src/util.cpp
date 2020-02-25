#include "util.hpp"
#include <chrono>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <random>
#include <thread>

namespace vllm
{
// L2 norm is used
pcl::CorrespondencesPtr getCorrespondences(const pcXYZ::Ptr& cloud_source, const pcXYZ::Ptr& cloud_target)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source, target;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
  est.setInputSource(cloud_source);
  est.setInputTarget(cloud_target);
  pcl::CorrespondencesPtr all_correspondences(new pcl::Correspondences);
  est.determineCorrespondences(*all_correspondences);

  return all_correspondences;
}

// get scale factor from rotation matrix
double getScale(const Eigen::Matrix3f& R)
{
  return std::sqrt((R.transpose() * R).trace() / 3.0);
}

// Not compatible with Scaling
Eigen::Matrix4f icpWithPurePCL(const pcXYZ::Ptr& cloud_query, const pcXYZ::Ptr& cloud_reference)
{
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  pcXYZ::Ptr cloud_aligned(new pcXYZ);
  Eigen::Matrix4f T = icp.getFinalTransformation();

  icp.setInputSource(cloud_query);
  icp.setInputTarget(cloud_reference);
  icp.align(*cloud_aligned);

  return T;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadMapPointCloud(const std::string& pcd_file, float leaf)
{
  // Load map pointcloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_map);

  std::cout << "leaf size=" << leaf << std::endl;
  if (leaf < 0) {
    return cloud_map;
  }
  // filtering
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud_map);
  filter.setLeafSize(leaf, leaf, leaf);
  filter.filter(*cloud_map);
  return cloud_map;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& pcd_file)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_source) == -1) {
    std::cout << "Couldn't read file test_pcd.pcd " << pcd_file << std::endl;
    exit(1);
  }
  return cloud_source;
}


pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(const pcXYZ::Ptr& cloud, float leaf)
{
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(leaf);
  ne.compute(*normals);

  return normals;
}

// Eigen::Matrix4f registrationPointCloud(
//     const pcXYZ::Ptr& cloud_source,
//     const pcXYZ::Ptr& cloud_target)
// {
//   using pclSVD = pcl::registration::TransformationEstimationSVDScale<pcl::PointXYZ, pcl::PointXYZ>;
//   pclSVD::Ptr estPtr(new pclSVD());

//   Eigen::Matrix4f T;
//   estPtr->estimateRigidTransformation(*cloud_source, *cloud_target, getCorrespondences(cloud_source, cloud_target), T);
//   Eigen::Matrix3f R = T.topLeftCorner(3, 3);

//   return T;
// }

void shufflePointCloud(pcXYZ::Ptr& cloud)
{
  std::mt19937 rand;
  for (size_t i = 0, size = cloud->size(); i < size; i++) {
    std::swap(cloud->points.at(i), cloud->points.at(rand() % size));
  }
}

void wait(float ms)
{
  std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(ms * 1000.f)));
}

Eigen::Matrix3f randomRotation()
{
  return Eigen::Quaternionf::UnitRandom().toRotationMatrix();
}

}  // namespace vllm