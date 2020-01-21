#include "aligner.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <random>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

pcXYZ::Ptr loadPointCloud(const std::string& pcd_file)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
  // load point cloud
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_in) == -1) {
    std::cout << "Couldn't read file test_pcd.pcd " << pcd_file << std::endl;
    exit(1);
  }
  return cloud_in;
}

void transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  // rotation
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(static_cast<float>(M_PI / 6.0), Eigen::Vector3f(0, 1, 0));  // Rotate CCW 90[deg] around y-axis
  // R = Eigen::Matrix3f::Identity();

  // scaling
  float scale = 1.2f;
  R *= scale;

  // transration
  Eigen::Vector3f t;
  t << 0.05f, -0.1f, 0.0f;

  // transform
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = R;
  T.topRightCorner(3, 1) = t;

  std::cout << "\n=========== " << std::endl;
  std::cout << "scale " << scale << std::endl;
  std::cout << T << std::endl;
  pcl::transformPointCloud(*cloud, *cloud, T);
}

// L2 norm is maybe used
pcl::Correspondences getCorrespondences(const pcXYZ::Ptr& cloud_source, const pcXYZ::Ptr& cloud_target)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source, target;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
  est.setInputSource(cloud_source);
  est.setInputTarget(cloud_target);
  pcl::Correspondences all_correspondences;
  // est.determineReciprocalCorrespondences(all_correspondences);
  est.determineCorrespondences(all_correspondences);

  return all_correspondences;
}

double getScale(const Eigen::Matrix3f& R)
{
  return std::sqrt((R.transpose() * R).trace() / 3.0);
}

Eigen::Matrix4f registrationPointCloud(
    const pcXYZ::Ptr& cloud_source,
    const pcXYZ::Ptr& cloud_target)
{
  using pclSVD = pcl::registration::TransformationEstimationSVDScale<pcl::PointXYZ, pcl::PointXYZ>;
  pclSVD::Ptr estPtr(new pclSVD());

  Eigen::Matrix4f T;
  estPtr->estimateRigidTransformation(*cloud_source, *cloud_target, getCorrespondences(cloud_source, cloud_target), T);
  Eigen::Matrix3f R = T.topLeftCorner(3, 3);
  std::cout << "\n=========== " << std::endl;
  std::cout << "scale " << getScale(R) << std::endl;
  std::cout << T << std::endl;

  return T;
}

Eigen::Matrix4f registrationByG2O(
    const pcXYZ::Ptr& cloud_source,
    const pcXYZ::Ptr& cloud_target,
    const pcl::Correspondences& correspondences)
{
  vllm::Aligner aligner;
  Eigen::Matrix4f T;
  aligner.estimate(
      *cloud_source, *cloud_target, correspondences, T);
  Eigen::Matrix3f R = T.topLeftCorner(3, 3);
  std::cout << "\n=========== " << std::endl;
  std::cout << "scale " << getScale(R) << std::endl;
  std::cout << T << std::endl;

  return T;
}


// Not compatible with Scaling
Eigen::Matrix4f icpWithPurePCL(const pcXYZ::Ptr& cloud_query, const pcXYZ::Ptr& cloud_reference)
{
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  pcXYZ::Ptr cloud_aligned(new pcXYZ);
  Eigen::Matrix4f T = icp.getFinalTransformation();

  icp.setInputSource(cloud_query);
  icp.setInputTarget(cloud_reference);
  icp.align(*cloud_aligned);
  std::cout << T << std::endl;

  return T;
}

void shufflePointCloud(pcXYZ::Ptr& cloud)
{
  std::mt19937 rand;
  for (size_t i = 0, size = cloud->size(); i < size; i++) {
    std::swap(cloud->points.at(i), cloud->points.at(rand() % size));
  }
}

bool quit = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void*)
{
  if (event.getKeySym() == "q" && event.keyDown()) {
    std::cout << "pressed" << std::endl;
    quit = true;
  }
}

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("visualizer"));

void visualizePointCloud(
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
  // viewer->addCorrespondences<pcl::PointXYZ>(cloud_aligned, cloud_target, correspondences);

  // while (!(viewer->wasStopped() || quit)) {
  viewer->spinOnce(500);
  // }
}

int main(int, char**)
{
  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = loadPointCloud("../data/table.pcd");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_target = *cloud_source;

  transformPointCloud(cloud_target);
  shufflePointCloud(cloud_source);

  viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_align(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_align = *cloud_source;

  for (int i = 0; i < 100; i++) {
    pcl::Correspondences correspondences = getCorrespondences(cloud_align, cloud_target);

    Eigen::Matrix4f T = registrationByG2O(cloud_align, cloud_target, correspondences);

    pcl::transformPointCloud(*cloud_align, *cloud_align, T);

    visualizePointCloud(cloud_source, cloud_target, cloud_align, correspondences);
  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(1000);
  }
  return 0;
}