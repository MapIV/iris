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


void Viewer::visualizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned)
{
  pcl::Correspondences correspondences;
  visualizePointCloud(cloud_source, cloud_target, cloud_aligned, correspondences);
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
  viewer->addPointCloud<pcl::PointXYZ>(cloud_source, pcl_color(cloud_source, 255, 0, 0), "cloud_source");
  viewer->addPointCloud<pcl::PointXYZ>(cloud_target, pcl_color(cloud_target, 0, 255, 0), "cloud_target");
  viewer->addPointCloud<pcl::PointXYZ>(cloud_aligned, pcl_color(cloud_aligned, 0, 0, 255), "cloud_aligned");
  viewer->addCorrespondences<pcl::PointXYZ>(cloud_aligned, cloud_target, correspondences);
}
}  // namespace vllm