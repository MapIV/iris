#include "core/util.hpp"
#include <chrono>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <random>
#include <thread>

namespace vllm
{

namespace
{
float getScale_(const Eigen::MatrixXf& R)
{
  return static_cast<float>(std::sqrt((R.transpose() * R).trace() / 3.0));
}

// return the closest rotatin matrix
Eigen::Matrix3f normalize_(const Eigen::Matrix3f& R)
{
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f Vt = svd.matrixV().transpose();
  if (R.determinant() < 0) {
    return -U * Vt;
  }

  return U * Vt;
}

}  // namespace

// get scale factor from rotation matrix
float getScale(const Eigen::MatrixXf& A)
{
  if (A.cols() == 3)
    return getScale_(A);
  else if (A.cols() == 4)
    return getScale_(A.topLeftCorner(3, 3));
  return -1;
}


Eigen::Matrix3f normalizeRotation(const Eigen::MatrixXf& A)
{
  if (A.cols() != 3 && A.cols() != 4) {
    exit(1);
  }

  Eigen::Matrix3f sR = A.topLeftCorner(3, 3);
  float scale = getScale(sR);
  return normalize_(sR / scale);
}

Eigen::Matrix4f normalizePose(const Eigen::Matrix4f& sT)
{
  Eigen::Matrix4f T = sT;
  T.topLeftCorner(3, 3) = normalizeRotation(sT);
  return T;
}

void loadMap(
    const std::string& pcd_file,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    float grid_leaf, float radius)
{
  cloud->clear();
  normals->clear();

  // Load map pointcloud
  pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud);

  {
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(cloud);
    Eigen::Vector4f min4, max4;
    min4 << -5, -5, -5, 1;
    max4 << 5, 5, 5, 1;
    crop.setMin(min4);
    crop.setMax(max4);
    crop.filter(*cloud);
  }

  // normal estimation
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(*normals);

  if (grid_leaf < 0) return;

  // filtering
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud);
  filter.setLeafSize(grid_leaf, grid_leaf, grid_leaf);
  filter.filter(*cloud);
  pcl::IndicesPtr ind = filter.getIndices();

  pcl::PointCloud<pcl::Normal>::Ptr tmp_normals(new pcl::PointCloud<pcl::Normal>());
  for (int i : *ind) {
    tmp_normals->push_back(normals->at(i));
  }
  std::cout << "normals " << normals->size() << " tmp_normals " << tmp_normals->size() << " points " << cloud->size() << " ind " << ind->size() << std::endl;
  std::cout << "removed_ind " << filter.getRemovedIndices()->size() << std::endl;

  normals = tmp_normals;

  return;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadMapPointCloud(const std::string& pcd_file, float leaf)
{
  // Load map pointcloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_map);

  // {
  //   pcl::CropBox<pcl::PointXYZ> crop;
  //   crop.setInputCloud(cloud_map);
  //   Eigen::Vector4f min4, max4;
  //   min4 << -5, -5, -5, 1;
  //   max4 << 5, 5, 5, 1;
  //   crop.setMin(min4);
  //   crop.setMax(max4);
  //   crop.filter(*cloud_map);
  // }

  if (leaf < 0) {
    return cloud_map;
  }

  // filtering
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud_map);
  filter.setLeafSize(leaf, leaf, leaf);
  filter.filter(*cloud_map);
  return cloud_map;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& pcd_file)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_source) == -1) {
    std::cout << "Couldn't read file test_pcd.pcd " << pcd_file << std::endl;
    exit(1);
  }
  return cloud_source;
}


pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(const pcXYZ::Ptr& cloud, float leaf)
{
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(leaf);
  ne.compute(*normals);

  return normals;
}

void transformNormals(const pcNormal& source, pcNormal& target, const Eigen::Matrix4f& T)
{
  Eigen::Matrix3f R = normalizeRotation(T);
  if (&source != &target) {
    target.clear();
    for (const pcl::Normal& n : source) {
      Eigen::Vector3f _n = R * n.getNormalVector3fMap();
      target.push_back({_n.x(), _n.y(), _n.z()});
    }
    return;
  }

  for (pcl::Normal& n : target) {
    Eigen::Vector3f _n = R * n.getNormalVector3fMap();
    n = pcl::Normal(_n.x(), _n.y(), _n.z());
  }
  return;
}

Eigen::Matrix3f randomRotation()
{
  return Eigen::Quaternionf::UnitRandom().toRotationMatrix();
}

void shufflePointCloud(pcXYZ::Ptr& cloud)
{
  std::mt19937 rand;
  for (size_t i = 0, size = cloud->size(); i < size; i++) {
    std::swap(cloud->points.at(i), cloud->points.at(rand() % size));
  }
}

}  // namespace vllm