#include "aligner.hpp"
#include "util.hpp"
#include <pcl/common/transforms.h>
#include <pcl/registration/correspondence_rejection_distance.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

int main(/*int argc, char* argv[]*/)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = vllm::loadMapPointCloud("../data/table.pcd", 0.1f);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::cout << "target size =" << target_cloud->size() << std::endl;

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  Eigen::Vector3f t;
  Eigen::Matrix3f R = Eigen::AngleAxisf(10.f * 3.14f / 180.f, Eigen::Vector3f::UnitX()).matrix();
  t << 0, 0.05f, 0.1f;
  T.topRightCorner(3, 1) = t;
  T.topLeftCorner(3, 3) = R;

  std::cout << "opt=\n"
            << T << std::endl;
  pcl::transformPointCloud(*target_cloud, *source_cloud, T.inverse());

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();

  for (int i = 0; i < 20; i++) {
    pcl::transformPointCloud(*source_cloud, *aligned_cloud, T_align);
    pcl::CorrespondencesPtr correspondences = vllm::getCorrespondences(aligned_cloud, target_cloud);
    std::cout << "correspondences size=" << correspondences->size() << std::endl;

    vllm::Aligner aligner;
    aligner.setGain(1e4, 1e4);
    T_align = aligner.estimate6DoF(T_align, *source_cloud, *target_cloud, *correspondences);

    std::cout << T_align << std::endl;
  }

  return 0;
}
