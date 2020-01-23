#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
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

// local point distribution
struct LPD {
  size_t N;
  Eigen::Matrix4f T;
  Eigen::Vector3f sigma;
};

float getLongestRange(pcl::PointXYZ min, pcl::PointXYZ max)
{
  float x = max.x - min.x;
  float y = max.y - min.y;
  float z = max.z - min.z;
  return std::max(x, std::max(y, z));
}

int main(int, char**)
{
  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = loadPointCloud("../data/table.pcd");

  pcl::PointXYZ min_point, max_point;
  pcl::getMinMax3D(*cloud_source, min_point, max_point);
  constexpr int N = 2;  // division
  const float L = getLongestRange(min_point, max_point);
  const float D = L / N;  // resolution

  Eigen::Vector4f bottom;
  bottom << min_point.x, min_point.y, min_point.z, 1.0f;

  pcl::CropBox<pcl::PointXYZ> clop;
  clop.setInputCloud(cloud_source);

  pcl::PointCloud<pcl::PointXYZ> cloud_cropped;
  std::cout << "source " << cloud_source->size() << std::endl;

  // cut out pointcloud for each voxel
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        Eigen::Vector4f min_box, max_box;
        min_box << D * static_cast<float>(i), D * static_cast<float>(j), D * static_cast<float>(k), 1.0f;
        max_box << D * static_cast<float>(i + 1), D * static_cast<float>(j + 1), D * static_cast<float>(k + 1), 1.0f;
        clop.setMax(bottom + max_box);
        clop.setMin(bottom + min_box);
        clop.filter(cloud_cropped);
        std::cout << cloud_cropped.size() << std::endl;
      }
    }
  }
  return 0;
}