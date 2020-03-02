#pragma once
#include "bridge.hpp"
#include "config.hpp"
#include "rejector_lpd.hpp"
#include "util.hpp"
#include <memory>
#include <pcl/registration/correspondence_rejection_distance.h>

namespace vllm
{
using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

class System
{
public:
  System(int argc, char* argv[]);

  int execute();

private:
  Config config;

  // setup for others
  bool vllm_pause = false;
  Eigen::Matrix4f T_init;
  Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
  std::vector<Eigen::Vector3f> raw_trajectory;
  std::vector<Eigen::Vector3f> vllm_trajectory;
  Eigen::Matrix4f raw_camera = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f vllm_camera = Eigen::Matrix4f::Identity();

  // Rejector
  vllm::GPD gpd;
  vllm::CorrespondenceRejectorLpd lpd_rejector;
  pcl::registration::CorrespondenceRejectorDistance distance_rejector;

  BridgeOpenVSLAM bridge;

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr aligned_normals;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr source_normals;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr target_normals;
};

}  // namespace vllm