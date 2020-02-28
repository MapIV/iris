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
  PangolinViewer(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const pangolin::AxisDirection up);
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

  // void visualize()
  // {
  //   pangolin_viewer.clear();
  //   pangolin_viewer.drawGridLine();
  //   pangolin_viewer.drawString("VLLM", {1.0f, 1.0f, 0.0f, 3.0f});

  //   pangolin_viewer.drawPointCloud(aligned_cloud, {1.0f, 1.0f, 0.0f, 2.0f});
  //   pangolin_viewer.drawPointCloud(target_cloud, {0.6f, 0.6f, 0.6f, 1.0f});
  //   pangolin_viewer.drawTrajectory(vllm_trajectory, {1.0f, 0.0f, 0.0f, 3.0f});
  //   pangolin_viewer.drawCamera(vllm_camera, {1.0f, 0.0f, 0.0f, 1.0f});
  //   // pangolin_viewer.drawCorrespondences(aligned_cloud, target_cloud, correspondences, {0.0f, 1.0f, 0.0f, 2.0f});

  //   if (gui_raw_camera->value()) {
  //     pangolin_viewer.drawCamera(raw_camera, {1.0f, 0.0f, 1.0f, 1.0f});
  //     pangolin_viewer.drawTrajectory(raw_trajectory, {1.0f, 0.0f, 1.0f, 3.0f});
  //   }
  //   if (gui_source_normals->value())
  //     pangolin_viewer.drawNormals(aligned_cloud, aligned_normals, {1.0f, 0.0f, 1.0f, 1.0f});
  //   if (gui_target_normals->value())
  //     pangolin_viewer.drawNormals(target_cloud, target_normals, {0.0f, 1.0f, 1.0f, 1.0f}, 30);
  //   if (gui_gpd->value())
  //     pangolin_viewer.drawGPD(gpd);

  //   pangolin_viewer.swap();
  // }

  void drawGPD(const GPD& gpd) const;
  void drawGridLine() const;
  void drawString(const std::string& str, const Color& color) const;
  void drawTrajectory(const std::vector<Eigen::Vector3f>& trajectory, const Color& color);
  void drawPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Color& color) const;
  void drawCamera(const Eigen::Matrix4f& cam_pose, const Color& color) const;
  void drawNormals(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals,
      const Color& color,
      int skip = 1) const;
  void drawCorrespondences(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::CorrespondencesPtr& correspondences,
      const Color& color) const;

private:
  pangolin::OpenGlRenderState s_cam;
  pangolin::Handler3D handler;
  pangolin::View d_cam;

  // GUI variables
  std::shared_ptr<pangolin::Var<bool>> gui_raw_camera;
  std::shared_ptr<pangolin::Var<bool>> gui_source_normals;
  std::shared_ptr<pangolin::Var<bool>> gui_target_normals;
  std::shared_ptr<pangolin::Var<bool>> gui_gpd;

  void drawRectangular(const float x, const float y, const float z) const;
  void drawLine(
      const float x1, const float y1, const float z1,
      const float x2, const float y2, const float z2) const;

  void drawFrustum(const float w) const;
};
}  // namespace vllm