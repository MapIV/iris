#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcNormal = pcl::PointCloud<pcl::Normal>;
}  // namespace vllm