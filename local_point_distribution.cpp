#include "lpd.hpp"
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

void transformPointCloud(pcXYZ::Ptr& cloud_ptr)
{
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(static_cast<float>(M_PI / 6.0), Eigen::Vector3f(0, 0, 1));  // Rotate CCW 90[deg] around z-axis
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = R;
  pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, T);
}


namespace vllm
{
namespace CorrespondenceRejectorLpd
{
const int N = 2;

LPD table[N][N][N];

// blok size
pcl::PointXYZ D;

pcl::PointXYZ getResolution(pcl::PointXYZ min, pcl::PointXYZ max, int N)
{
  float inv = 1.0f / static_cast<float>(N);
  return pcl::PointXYZ((max.x - min.x) * inv, (max.y - min.y) * inv, (max.z - min.z) * inv);
}

void init(pcXYZ::Ptr cloud)
{
  pcl::PointXYZ min_point, max_point;
  pcl::getMinMax3D(*cloud, min_point, max_point);

  D = getResolution(min_point, max_point, N);

  Eigen::Vector4f bottom;
  bottom << min_point.x, min_point.y, min_point.z, 1.0f;

  pcl::CropBox<pcl::PointXYZ> clop;
  clop.setInputCloud(cloud);

  LpdAnalyzer analyzer;

  // cut out pointcloud for each voxel
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        Eigen::Vector4f min_box, max_box;
        min_box << D.x * static_cast<float>(i), D.y * static_cast<float>(j), D.z * static_cast<float>(k), 1.0f;
        max_box << D.x * static_cast<float>(i + 1), D.y * static_cast<float>(j + 1), D.z * static_cast<float>(k + 1), 1.0f;

        pcXYZ cloud_cropped;
        clop.setMax(bottom + max_box);
        clop.setMin(bottom + min_box);
        clop.filter(cloud_cropped);

        LPD tmp = analyzer.compute(cloud_cropped.makeShared());
        table[i][j][k] = tmp;
        tmp.show();
      }
    }
  }
}

}  // namespace CorrespondenceRejectorLpd
}  // namespace vllm

int main(int, char**)
{
  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = loadPointCloud("../data/table.pcd");

  transformPointCloud(cloud_source);
  std::cout << cloud_source->size() << std::endl;

  vllm::CorrespondenceRejectorLpd::init(cloud_source);

  return 0;
}