#pragma once
#include "global_point_distribution.hpp"
#include "pangolin_cloud.hpp"
#include <pangolin/pangolin.h>
#include <pcl/correspondence.h>

namespace vllm
{
struct Color {
  float r;
  float g;
  float b;
  float size;
  Color() { r = g = b = size = 1.0f; }
  Color(float r, float g, float b, float s) : r(r), g(g), b(b), size(s) {}
};

class PangolinViewer
{
public:
  PangolinViewer();
  ~PangolinViewer() = default;

  void swap() const
  {
    pangolin::FinishFrame();
  }

  void clear()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
  }

  void drawGPD(const GPD& gpd) const;
  void drawGridLine() const;
  void drawString(const std::string& str, const Color& color) const;

  void drawPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Color& color) const;
  void drawCamera(const Eigen::Matrix4f& cam_pose, const Color& color) const;
  void drawNormals(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals,
      const Color& color) const;
  void drawCorrespondences(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::Correspondences& correspondences,
      const Color& color) const;

private:
  std::shared_ptr<pangolin::Var<double>> ui_double_ptr;
  pangolin::OpenGlRenderState s_cam;
  pangolin::Handler3D handler;
  pangolin::View d_cam;

  void drawRectangular(const float x, const float y, const float z) const;
  void drawLine(
      const float x1, const float y1, const float z1,
      const float x2, const float y2, const float z2) const;

  void drawFrustum(const float w) const;
};
}  // namespace vllm