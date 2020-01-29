#pragma once
#include "global_point_distribution.hpp"
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>

namespace vllm
{
class Viewer
{
private:
  unsigned char key = 0;
  pcl::visualization::PCLVisualizer::Ptr viewer;

public:
  Viewer();

  void visualizePointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source);

  void visualizePointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned);

  void visualizePointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_aligned,
      const pcl::Correspondences& correspondences);

  void visualizeGPD(const GPD& gpd);

  int milliSeconds(std::chrono::system_clock::time_point start)
  {
    auto dur = std::chrono::system_clock::now() - start;
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count());
  }

  int waitKey(int ms = -1)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    viewer->spinOnce(1, true);
    key = 0;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    while (key == 0) {
      if (milliSeconds(start) > ms && ms > 0) return -1;
      viewer->spinOnce(1, true);
    }
    std::cout << key << " " << int(key) << std::endl;
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