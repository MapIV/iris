#include "aligner.hpp"
#include "rejector_lpd.hpp"
#include "util.hpp"
#include "viewer.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

pcXYZ::Ptr loadPointCloud(const std::string& pcd_file)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_source) == -1) {
    std::cout << "Couldn't read file test_pcd.pcd " << pcd_file << std::endl;
    exit(1);
  }
  return cloud_source;
}

void initTransformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
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

  // std::cout << "\n=========== " << std::endl;
  // std::cout << "scale " << scale << std::endl;
  // std::cout << T << std::endl;
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
  // std::cout << "\n=========== " << std::endl;
  // std::cout << "scale " << vllm::getScale(R) << std::endl;
  // std::cout << T << std::endl;

  return T;
}

int main(int argc, char** argv)
{
  float gain = 1.0f;
  if (argc == 2)
    gain = static_cast<float>(std::atof(argv[1]));

  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = loadPointCloud("../data/table.pcd");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_target = *cloud_source;
  initTransformPointCloud(cloud_target);

  vllm::Viewer viewer;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_align(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_align = *cloud_source;


  // viewer.visualizePointCloud(cloud_source, cloud_target, cloud_align);
  viewer.visualizePointCloud(cloud_target);

  vllm::GPD gpd(5);
  gpd.init(cloud_target);
  viewer.visualizeGPD(gpd);


  while (viewer.waitKey() != 's')
    ;

  vllm::CorrespondenceRejectorLpd rejector(gpd, gain);
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  for (int i = 0; i < 100; i++) {
    pcl::Correspondences correspondences = getCorrespondences(cloud_align, cloud_target);
    viewer.visualizePointCloud(cloud_source, cloud_target, cloud_align, correspondences);
    std::cout << "NN" << std::endl;
    if (viewer.waitKey(0) == 'q') break;

    // refine
    pcl::Correspondences refined = rejector.refineCorrespondences(correspondences, cloud_align);
    viewer.visualizePointCloud(cloud_source, cloud_target, cloud_align, refined);
    std::cout << "all: " << correspondences.size() << " refine: " << refined.size() << std::endl;
    if (viewer.waitKey(0) == 'q') break;

    Eigen::Matrix4f T = registrationByG2O(cloud_align, cloud_target, refined);
    pcl::transformPointCloud(*cloud_align, *cloud_align, T);
    pose = T * pose;

    viewer.visualizePointCloud(cloud_source, cloud_target, cloud_align);
    std::cout << "trans" << std::endl;
    if (viewer.waitKey(0) == 'q') break;
  }
  std::cout << pose << std::endl;
  return 0;
}