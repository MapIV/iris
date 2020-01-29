#include "viewer.hpp"

namespace vllm
{
namespace
{
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* code_void)
{
  unsigned char* key = static_cast<unsigned char*>(code_void);
  *key = event.getKeyCode();
}
}  // namespace

Viewer::Viewer()
{
  viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("visualizer"));
  viewer->registerKeyboardCallback(keyboardEventOccurred, &key);
}

void Viewer::visualizeGPD(const GPD& gpd)
{
  const size_t N = gpd.N;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        LPD lpd = gpd.data[i][j][k];
        if (lpd.N < 50) continue;
        Eigen::Quaternionf q(lpd.R());
        viewer->addCube(lpd.t(), q, lpd.sigma.x(), lpd.sigma.y(), lpd.sigma.z(), "cube" + std::to_string(i + j * N + k * N * N));
      }
    }
  }
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        Eigen::Vector3f p = gpd.bottom;
        const Eigen::Vector3f& s = gpd.segment;
        p.x() += i * s.x();
        p.y() += j * s.y();
        p.z() += k * s.z();
        viewer->addSphere(pcl::PointXYZ(p.x(), p.y(), p.z()), 0.01, 0, 255, 0, "sphere" + std::to_string(i + j * N + k * N * N));
      }
    }
  }
}

void Viewer::visualizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned)
{
  pcl::Correspondences correspondences;
  visualizePointCloud(cloud_source, cloud_target, cloud_aligned, correspondences);
}

void Viewer::visualizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source)
{
  using pcl_color = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>;
  viewer->removeAllPointClouds();
  viewer->removeCorrespondences();
  viewer->addPointCloud<pcl::PointXYZ>(cloud_source, pcl_color(cloud_source, 255, 0, 255), "cloud_source");
}


void Viewer::visualizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned,
    const pcl::Correspondences& correspondences)
{
  using pcl_color = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>;
  viewer->removeAllPointClouds();
  viewer->removeCorrespondences();
  viewer->addPointCloud<pcl::PointXYZ>(cloud_source, pcl_color(cloud_source, 255, 0, 255), "cloud_source");
  viewer->addPointCloud<pcl::PointXYZ>(cloud_target, pcl_color(cloud_target, 0, 255, 255), "cloud_target");
  viewer->addPointCloud<pcl::PointXYZ>(cloud_aligned, pcl_color(cloud_aligned, 255, 255, 255), "cloud_aligned");
  viewer->addCorrespondences<pcl::PointXYZ>(cloud_aligned, cloud_target, correspondences);
}
}  // namespace vllm