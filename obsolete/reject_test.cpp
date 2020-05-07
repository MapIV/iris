#include "rejector_lpd.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

constexpr int N = 50;
constexpr float PI = 3.14159f;
constexpr float OMEGA = PI / N;

pcXYZ makeSphere()
{
  const float R = 1.0f;
  pcXYZ cloud;
  cloud.points.resize(2 * N * N);
  cloud.width = 2 * N * N;
  cloud.height = 1;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < 2 * N; j++) {
      float x = R * std::cos(OMEGA * i);
      float y = R * std::sin(OMEGA * i) * std::cos(OMEGA * j);
      float z = R * std::sin(OMEGA * i) * std::sin(OMEGA * j);
      cloud.points.at(i + N * j) = pcl::PointXYZ(x, y, z);
    }
  }

  return cloud;
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
  pcl::PointCloud<pcl::PointXYZ> cloud = makeSphere();
  pcl::io::savePCDFileBinary("sphere.pcd", cloud);

  vllm::CorrespondenceRejectorLpd rejector;
  rejector.init(cloud.makeShared());

  //clang-format off
  std::vector<pcl::PointXYZ> points{
      {0, 0, 0}, {1, 1, 1},
      {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
      {1, 1, 0}, {1, 0, 1}, {0, 1, 1},
      {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
      {3, 0, 0}, {0, 3, 0}, {0, 0, 3}};
  //clang-format on

  for (int i = 0; i < points.size(); i++) {
    bool flag = rejector.check(points.at(i));
    std::cout << std::boolalpha << flag << std::endl;
  }

  return 0;
}