#pragma once
#include "pcl_/correspondence_estimator.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace iris
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcNormal = pcl::PointCloud<pcl::Normal>;
using pcXYZIN = pcl::PointCloud<pcl::PointXYZINormal>;
using xyzin = pcl::PointXYZINormal;
using crrspEstimator = iris::pcl_::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal>;
using crrspRejector = pcl::registration::CorrespondenceRejectorDistance;

}  // namespace iris