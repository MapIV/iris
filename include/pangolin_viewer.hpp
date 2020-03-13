#pragma once
#include "color.hpp"
#include "global_point_distribution.hpp"
#include "pangolin_cloud.hpp"
#include "system.hpp"
#include <atomic>
#include <pangolin/pangolin.h>
#include <pcl/correspondence.h>
#include <thread>

namespace vllm
{
class PangolinViewer
{
public:
  // constructor
  PangolinViewer(const std::shared_ptr<System>& system_ptr);
  PangolinViewer() : PangolinViewer(nullptr){};

  // destructor
  ~PangolinViewer() = default;

  // initialize
  void init();

  // beggining and finishing
  void clear();
  void swap();

  void execute();
  void startLoop();
  void quitLoop();

  // draw functions
  void drawGridLine() const;
  void drawString(const std::string& str, const Color& color) const;
  void drawTrajectory(const std::vector<Eigen::Vector3f>& trajectory, bool colorful, const Color& color = Color());
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
  std::shared_ptr<System> system_ptr = nullptr;
  std::shared_ptr<pangolin::OpenGlRenderState> s_cam;
  std::shared_ptr<pangolin::Handler3D> handler;
  pangolin::View d_cam;

  std::thread viewer_thread;
  std::atomic<bool> loop_flag;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr target_normals;

  pangolin::OpenGlRenderState makeCamera(
      const Eigen::Vector3f& from = Eigen::Vector3f(-2, 0, 5),
      const Eigen::Vector3f& to = Eigen::Vector3f(0, 0, 0),
      const pangolin::AxisDirection up = pangolin::AxisX);

  void loop();

  // GUI variables
  std::shared_ptr<pangolin::Var<bool>> gui_vslam_camera;
  std::shared_ptr<pangolin::Var<bool>> gui_source_normals;
  std::shared_ptr<pangolin::Var<bool>> gui_target_normals;
  std::shared_ptr<pangolin::Var<double>> gui_scale_gain;
  std::shared_ptr<pangolin::Var<double>> gui_pitch_gain;
  std::shared_ptr<pangolin::Var<double>> gui_model_gain;
  std::shared_ptr<pangolin::Var<double>> gui_distance_min;
  std::shared_ptr<pangolin::Var<double>> gui_distance_max;
  std::shared_ptr<pangolin::Var<unsigned int>> gui_recollection;
  std::shared_ptr<pangolin::Var<bool>> gui_quit;
  std::shared_ptr<pangolin::Var<bool>> gui_reset;

  void drawFrustum(const float w) const;
  void drawRectangular(const float x, const float y, const float z) const;
  void drawLine(
      const float x1, const float y1, const float z1,
      const float x2, const float y2, const float z2) const;

  // h[0,360],s[0,1],v[0,1]
  Eigen::Vector3f convertRGB(Eigen::Vector3f hsv);
};
}  // namespace vllm
