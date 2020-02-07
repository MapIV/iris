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

// L2 norm is maybe used
pcl::Correspondences getCorrespondences(const pcXYZ::Ptr& cloud_source, const pcXYZ::Ptr& cloud_target)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source, target;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
  est.setInputSource(cloud_source);
  est.setInputTarget(cloud_target);
  pcl::Correspondences all_correspondences;
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
  return T;
}

int main(int argc, char** argv)
{
  float gain = 0.4f;
  if (argc == 2)
    gain = static_cast<float>(std::atof(argv[1]));

  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source = loadPointCloud("../data/table.pcd");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_target = *cloud_source;
  initTransformPointCloud(cloud_source);


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_align(new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_align = *cloud_source;

  vllm::Viewer viewer;
  viewer.addPointCloud(cloud_target, "target", vllm::Color(255, 0, 0));
  viewer.addPointCloud(cloud_align, "align", vllm::Color(0, 255, 0));
  while (viewer.waitKey() != 's')
    ;

  vllm::GPD gpd(4);
  gpd.init(cloud_target, gain);
  viewer.visualizeGPD(gpd);
  vllm::CorrespondenceRejectorLpd rejector(gpd);

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  const int DT = 50;
  for (int i = 0; i < 100; i++) {
    // NN
    pcl::Correspondences correspondences = getCorrespondences(cloud_align, cloud_target);
    viewer.updatePointCloud(cloud_align, "align", vllm::Color(0, 255, 0));
    viewer.visualizeCorrespondences(cloud_align, cloud_target, correspondences, "cor", vllm::Color(0, 0, 255));
    if (viewer.waitKey(2 * DT) == 'q') break;

    // Rejector
    correspondences = rejector.refineCorrespondences(correspondences, cloud_align);
    viewer.updatePointCloud(cloud_align, "align", vllm::Color(0, 255, 0));
    viewer.visualizeCorrespondences(cloud_align, cloud_target, correspondences, "cor", vllm::Color(0, 255, 255));
    if (viewer.waitKey(2 * DT) == 'q') break;

    // Registration
    Eigen::Matrix4f T = registrationByG2O(cloud_align, cloud_target, correspondences);
    pcl::transformPointCloud(*cloud_align, *cloud_align, T);
    pose = T * pose;
    viewer.updatePointCloud(cloud_align, "align", vllm::Color(0, 255, 0));
    viewer.unvisualizeCorrespondences("cor");
    if (viewer.waitKey(DT) == 'q') break;

    std::cout << "trans " << T.topRightCorner(3, 1).transpose() << std::endl;
  }
  std::cout << pose << std::endl;
  return 0;
}