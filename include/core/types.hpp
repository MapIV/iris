#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/registration/correspondence_estimation_backprojection.h>
#include "estimator/correspondences_estiation_backprojection.hpp"
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcNormal = pcl::PointCloud<pcl::Normal>;
using crrspEstimator = pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal>;
using crrspRejector = pcl::registration::CorrespondenceRejectorDistance;

}  // namespace vllm