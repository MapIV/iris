#pragma once
#include "global_point_distribution.hpp"
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>

namespace vllm
{
struct Color {
  float r, g, b;
  Color() { r = g = b = 255; }
  Color(float r, float g, float b) : r(r), g(g), b(b) {}
};

class Viewer
{
private:
  unsigned char key = 0;
  pcl::visualization::PCLVisualizer::Ptr viewer;

public:
  Viewer();

  void addPointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string name,
      Color color = Color{}, double size = 3.0);
  void updatePointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string name,
      Color color = Color{}, double size = 3.0);

  void visualizeCorrespondences(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::Correspondences& correspondences,
      std::string name,
      Color color = Color{}, double size = 1.0);

  void unvisualizeCorrespondences(std::string name);

  void visualizeGPD(const GPD& gpd);


  int waitKey(int ms = -1)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    viewer->spinOnce(1, true);
    key = 0;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    while (key == 0) {
      if (milliSeconds(start) > ms && ms > 0) return -1;
      viewer->spinOnce(1, false);
    }
    // std::cout << key << " " << int(key) << std::endl;
    return key;
  }

private:
  int milliSeconds(std::chrono::system_clock::time_point start)
  {
    auto dur = std::chrono::system_clock::now() - start;
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count());
  }
};

}  // namespace vllm