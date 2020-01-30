#include "viewer.hpp"

namespace vllm
{
namespace
{
using pcl_color = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>;

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
  viewer->addCoordinateSystem(0.1);
}

void Viewer::visualizeGPD(const GPD& gpd)
{
  const size_t N = gpd.N;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        LPD lpd = gpd.data[i][j][k];
        if (lpd.N < 20) continue;
        Eigen::Quaternionf q(lpd.R());
        std::string name = "cube" + std::to_string(i + j * N + k * N * N);
        double gain = 1.0f / static_cast<double>(N);

        viewer->addCube(lpd.t(), q,
            2.0 * lpd.sigma.x(),
            2.0 * lpd.sigma.y(),
            2.0 * lpd.sigma.z(), name);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
            gain * static_cast<double>(i),
            gain * static_cast<double>(j),
            gain * static_cast<double>(k), name);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.02, name);
        // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, name);
      }
    }
  }
}

void Viewer::addPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string name, Color color, double size)
{
  viewer->addPointCloud<pcl::PointXYZ>(cloud, pcl_color(cloud, color.r, color.g, color.b), name);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, name);
}

void Viewer::updatePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string name, Color color, double size)
{
  viewer->updatePointCloud(cloud, pcl_color(cloud, color.r, color.g, color.b), name);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, name);
}

void Viewer::visualizeCorrespondences(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::Correspondences& correspondences,
    std::string name,
    Color color, double size)
{
  viewer->removeCorrespondences(name);
  viewer->addCorrespondences<pcl::PointXYZ>(source, target, correspondences, name);
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
      color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, name);
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, size, name);
}
void Viewer::unvisualizeCorrespondences(std::string name)
{
  viewer->removeCorrespondences(name);
}

}  // namespace vllm