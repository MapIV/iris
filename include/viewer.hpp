#pragma once
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>

namespace vllm
{
class Viewer
{
private:
  unsigned char key = false;
  pcl::visualization::PCLVisualizer::Ptr viewer;

public:
  Viewer();

  void visualizePointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned);

  void visualizePointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned,
      const pcl::Correspondences& correspondences);

  int milliSeconds(std::chrono::system_clock::time_point start)
  {
    auto dur = std::chrono::system_clock::now() - start;
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count());
  }

  unsigned char waitKey(int ms = -1)
  {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    key = 0;
    while (key == 0) {
      if (milliSeconds(start) > ms && ms > 0) return -1;

      viewer->spinOnce();
    }
    return key;
  }

  void watStop()
  {
    while (!viewer->wasStopped()) {
      viewer->spin();
    }
  }
};

}  // namespace vllm