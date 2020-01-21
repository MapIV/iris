#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <random>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

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
  std::cout << T << std::endl;

  return T;
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
//   std::cout << "\n=========== " << std::endl;
//   std::cout << "scale " << getScale(R) << std::endl;
//   std::cout << T << std::endl;

//   return T;
// }

void shufflePointCloud(pcXYZ::Ptr& cloud)
{
  std::mt19937 rand;
  for (size_t i = 0, size = cloud->size(); i < size; i++) {
    std::swap(cloud->points.at(i), cloud->points.at(rand() % size));
  }
}
}  // namespace vllm