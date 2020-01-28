#include "rejector_lpd.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

pcXYZ::Ptr loadPointCloud(const std::string& pcd_file)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_source) == -1) {
    std::cout << "Couldn't read file test_pcd.pcd " << pcd_file << std::endl;
    exit(1);
  }
  return cloud_source;
}

void transformPointCloud(pcXYZ::Ptr& cloud_ptr)
{
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(static_cast<float>(M_PI / 6.0), Eigen::Vector3f(0, 0, 1));  // Rotate CCW 90[deg] around z-axis
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = R;
  pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, T);
}

int main(int, char**)
{
  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = loadPointCloud("../data/table.pcd");

  transformPointCloud(cloud_source);

  vllm::CorrespondenceRejectorLpd rejector;
  rejector.init(cloud_source);

  for (int i = 0; i < 10; i++) {
    pcl::PointXYZ p = cloud_source->at(i);
    bool flag = rejector.check(p);
    std::cout << std::boolalpha << flag << std::endl;
  }
  bool flag = rejector.check(pcl::PointXYZ(0, 0, 0));
  std::cout << std::boolalpha << flag << std::endl;

  return 0;
}