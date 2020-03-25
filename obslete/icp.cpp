#include "aligner.hpp"
#include "pangolin_viewer.hpp"
#include "rejector_lpd.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

void initTransformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  // rotation
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(static_cast<float>(M_PI / 4.0), Eigen::Vector3f(0, 1, 0));  // Rotate CCW 90[deg] around y-axis
  // R = Eigen::Matrix3f::Identity();

  // scaling
  float scale = 1.5f;
  R *= scale;

  // transration
  Eigen::Vector3f t;
  t << 1.0f, -0.05f, 0.0f;

  // transform
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = R;
  T.topRightCorner(3, 1) = t;

  pcl::transformPointCloud(*cloud, *cloud, T);
}

int main(int argc, char** argv)
{
  float gain = 0.4f;
  if (argc == 2)
    gain = static_cast<float>(std::atof(argv[1]));

  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = vllm::loadPointCloud("../data/table.pcd");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_target = *cloud_source;
  initTransformPointCloud(cloud_source);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_align(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_align = *cloud_source;
  pcl::PointCloud<pcl::Normal>::Ptr normals = vllm::estimateNormals(cloud_target, 0.2f);

  // Initialize viewer
  vllm::PangolinViewer pangolin_viewer({0, 0.5f, 1.5f}, {0, 0, 0}, pangolin::AxisY);

  // Rejector
  vllm::GPD gpd(4);
  gpd.init(cloud_target, gain);
  vllm::CorrespondenceRejectorLpd rejector(gpd);

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

  for (int i = 0; i < 100; i++) {
    // Search Nearest Neighbor
    pcl::Correspondences correspondences = vllm::getCorrespondences(cloud_align, cloud_target);

    // Reject
    correspondences = rejector.refineCorrespondences(correspondences, cloud_align);
    // Align
    vllm::Aligner aligner;
    Eigen::Matrix4f T = aligner.estimate(*cloud_align, *cloud_target, correspondences, *normals);
    // Eigen::Matrix4f T = aligner.estimate(*cloud_align, *cloud_target, correspondences);

    pcl::transformPointCloud(*cloud_align, *cloud_align, T);
    pose = T * pose;
    std::cout << "itr=" << i << " t=" << T.topRightCorner(3, 1).transpose() << std::endl;

    pangolin_viewer.clear();
    pangolin_viewer.drawPointCloud(cloud_target, {1.0f, 1.0f, 0.0f, 1.0f});
    pangolin_viewer.drawPointCloud(cloud_align, {1.0f, 0.0f, 0.0f, 1.0f});
    pangolin_viewer.drawCorrespondences(cloud_align, cloud_target, correspondences, {1.0f, 1.0f, 1.0f, 1.0f});
    pangolin_viewer.drawNormals(cloud_target, normals, {0.0f, 0.0f, 1.0f, 2.0f});
    pangolin_viewer.drawString("iteration: " + std::to_string(i), {1.0f, 1.0f, 1.0f, 2.0f});
    pangolin_viewer.swap();
    vllm::wait(50);
  }
  std::cout << pose << std::endl;
  return 0;
}