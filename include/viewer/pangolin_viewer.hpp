#pragma once
#include "system/system.hpp"
#include "viewer/color.hpp"
#include "viewer/pangolin_cloud.hpp"
#include <atomic>
#include <mutex>
#include <pangolin/pangolin.h>
#include <pcl/correspondence.h>
#include <thread>

namespace vllm
{
namespace viewer
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

  // beggining and finishing for drawing
  void clear();
  void swap();

  // beggining and finishing for loop
  void execute();
  void startLoop();
  void quitLoop();

  void setIMU(const std::vector<Eigen::Matrix4f>& tra);

  bool isEnabledIMU() { return imu_use_flag.load(); }

private:
  void loop();

  // draw functions
  void drawFrustum(const float w) const;
  void drawRectangular(const float x, const float y, const float z) const;
  void drawLine(
      const float x1, const float y1, const float z1,
      const float x2, const float y2, const float z2) const;
  void drawPoses(const std::vector<Eigen::Matrix4f>& poses) const;
  void drawGridLine() const;
  void drawString(const std::string& str, const Color& color) const;
  void drawTrajectory(const std::vector<Eigen::Vector3f>& trajectory, bool colorful, const Color& color = Color());
  void drawPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Color& color) const;
  void drawPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, const Color& color) const;
  void drawCamera(const Eigen::Matrix4f& cam_pose, const Color& color) const;
  void drawNormals(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals,
      const Color& color,
      int skip = 1) const;
  void drawNormals(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const pcl::PointCloud<pcl::Normal>::Ptr& normals,
      const std::vector<Color>& colors,
      int skip = 1) const;
  void drawCorrespondences(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
      const pcl::CorrespondencesPtr& correspondences,
      const Color& color) const;

  pangolin::OpenGlRenderState makeCamera(
      const Eigen::Vector3f& from = Eigen::Vector3f(-10, 0, 10),
      const Eigen::Vector3f& to = Eigen::Vector3f(0, 0, 0),
      const pangolin::AxisDirection up = pangolin::AxisZ);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
  std::vector<Color> colorizeNormals(const pcl::PointCloud<pcl::Normal>::Ptr& normals);

  // private member
  std::shared_ptr<System> system_ptr = nullptr;
  std::shared_ptr<pangolin::OpenGlRenderState> s_cam;
  std::shared_ptr<pangolin::Handler3D> handler;
  pangolin::View d_cam;

  std::thread viewer_thread;
  std::atomic<bool> loop_flag;
  std::atomic<bool> imu_use_flag;

  std::mutex imu_mtx;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr target_normals;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored_target_cloud;
  std::vector<Color> target_normals_color;

  std::vector<Eigen::Matrix4f> imu_poses;

  Publication publication;

  map::Info localmap_info;

  // GUI variables
  std::shared_ptr<pangolin::Var<bool>> gui_vslam_camera;
  std::shared_ptr<pangolin::Var<bool>> gui_source_normals;
  std::shared_ptr<pangolin::Var<bool>> gui_target_normals;
  std::shared_ptr<pangolin::Var<bool>> gui_correspondences;

  std::shared_ptr<pangolin::Var<float>> gui_scale_gain;
  std::shared_ptr<pangolin::Var<float>> gui_smooth_gain;
  std::shared_ptr<pangolin::Var<float>> gui_latitude_gain;
  std::shared_ptr<pangolin::Var<float>> gui_altitude_gain;

  std::shared_ptr<pangolin::Var<float>> gui_distance_min;
  std::shared_ptr<pangolin::Var<float>> gui_distance_max;
  std::shared_ptr<pangolin::Var<unsigned int>> gui_recollection;
  std::shared_ptr<pangolin::Var<bool>> gui_quit;
  std::shared_ptr<pangolin::Var<bool>> gui_reset;
  std::shared_ptr<pangolin::Var<bool>> gui_imu;
};

}  // namespace viewer
}  // namespace vllm
