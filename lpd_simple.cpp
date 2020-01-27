#include "lpd.hpp"
#include <iostream>
#include <pcl/common/generate.h>
#include <pcl/common/random.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{
  // init point cloud
  //====================================
  // struct cloud generator
  float mean_x = 5, mean_y = 10, mean_z = 0;
  pcl::common::CloudGenerator<pcl::PointXYZ, pcl::common::NormalGenerator<float>> generator;
  std::uint32_t seed = static_cast<std::uint32_t>(time(nullptr));
  pcl::common::NormalGenerator<float>::Parameters x_params(mean_x, 1, seed++);
  pcl::common::NormalGenerator<float>::Parameters y_params(mean_y, 2, seed++);
  pcl::common::NormalGenerator<float>::Parameters z_params(mean_z, 0, seed++);
  generator.setParametersForX(x_params);
  generator.setParametersForY(y_params);
  generator.setParametersForZ(z_params);

  // set size
  int N = 100;
  if (argc == 2) {
    N = std::atoi(argv[1]);
  }
  std::cout << "N=" << N << std::endl;

  // generate cloud
  pcl::PointCloud<pcl::PointXYZ> cloud;
  generator.fill(N, 1, cloud);

  // transform
  {
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(static_cast<float>(M_PI / 6.0), Eigen::Vector3f(0, 0, 1));  // Rotate CCW 90[deg] around z-axis
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topLeftCorner(3, 3) = R;
    pcl::transformPointCloud(cloud, cloud, T);
  }

  // analyze Local Point Distribution
  //====================================
  vllm::LpdAnalyzer analyzer;
  vllm::LPD lpd = analyzer.compute(cloud.makeShared());
  lpd.show();

  pcl::io::savePCDFileBinary("random.pcd", cloud);
  return 0;
}