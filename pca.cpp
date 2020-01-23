#include <iostream>
#include <pcl/common/generate.h>
#include <pcl/common/pca.h>
#include <pcl/common/random.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

Eigen::Matrix3f correctRotationMatrix(const Eigen::Matrix3f& R)
{
  if (R.trace() < 0) {
    Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
    A(2, 2) = -1;
    return R * A;
  } else {
    return R;
  }
}

int main(int argc, char** argv)
{
  // struct cloud generator
  float mean = 0;
  pcl::common::CloudGenerator<pcl::PointXYZ, pcl::common::NormalGenerator<float>> generator;
  std::uint32_t seed = static_cast<std::uint32_t>(time(nullptr));
  pcl::common::NormalGenerator<float>::Parameters x_params(mean, 1, seed++);
  pcl::common::NormalGenerator<float>::Parameters y_params(mean, 2, seed++);
  pcl::common::NormalGenerator<float>::Parameters z_params(mean, 0, seed++);
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
    R = Eigen::AngleAxisf(static_cast<float>(M_PI / 6.0), Eigen::Vector3f(0, 1, 0));  // Rotate CCW 90[deg] around y-axis
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.topLeftCorner(3, 3) = R;
    pcl::transformPointCloud(cloud, cloud, T);
  }

  // primary component analysis
  {
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud.makeShared());
    Eigen::Matrix3f R = correctRotationMatrix(pca.getEigenVectors());
    Eigen::Vector3f sigma = pca.getEigenValues() / N;
    sigma = sigma.array().sqrt();

    std::cout << R << std::endl;
    std::cout << "" << std::endl;
    std::cout << sigma.transpose() << std::endl;
  }

  pcl::io::savePCDFileBinary("random.pcd", cloud);
  return 0;
}