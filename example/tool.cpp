#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;

pcXYZ::Ptr cropPointCloud(pcXYZ::Ptr source, Eigen::Vector3f min, Eigen::Vector3f max)
{
  pcXYZ::Ptr cropped(new pcXYZ);
  pcl::CropBox<pcl::PointXYZ> crop;

  Eigen::Vector4f min4 = Eigen::Vector4f::Ones();
  Eigen::Vector4f max4 = Eigen::Vector4f::Ones();
  min4.topRows(3) = min;
  max4.topRows(3) = max;

  crop.setInputCloud(source);
  crop.setMin(min4);
  crop.setMax(max4);
  crop.filter(*cropped);

  return cropped;
}

void setColor(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& gray,
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& color)
{
  for (size_t i = 0, N = gray->size(); i < N; i++) {
    pcl::PointXYZRGBA p;
    p.x = gray->at(i).x;
    p.y = gray->at(i).y;
    p.z = gray->at(i).z;
    p.b = 255;
    p.g = 255;
    uint8_t tmp = static_cast<uint8_t>(255 - std::min(std::abs(p.z) * 255.f / 10.0f, 200.f));
    p.a = p.r = tmp;
    color->push_back(p);
  }
}

struct Config {
  Config() {}

  Config(const std::string& yaml_file)
  {
    init(yaml_file);
  }

  void init(const std::string& yaml_file)
  {
    cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
    cv::Mat cv_max_box, cv_min_box, cv_normal;
    fs["Tool.max_box"] >> cv_max_box;
    fs["Tool.min_box"] >> cv_min_box;
    fs["Tool.normal"] >> cv_normal;
    fs["Tool.pcd_file"] >> pcd_file;
    fs["Tool.leaf_size"] >> leaf_size;
    fs["Tool.z_angle"] >> z_angle;

    cv::cv2eigen(cv_max_box, max_box);
    cv::cv2eigen(cv_min_box, min_box);
    cv::cv2eigen(cv_normal, normal);

    std::cout << "max " << max_box.transpose() << std::endl;
    std::cout << "min " << min_box.transpose() << std::endl;
    std::cout << "normal " << normal.transpose() << std::endl;
    std::cout << "leaf " << leaf_size << std::endl;
    std::cout << "z_angle " << z_angle << std::endl;

    T = Eigen::Matrix4f::Identity();

    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(normal, Eigen::Vector3f::UnitZ());
    T.topLeftCorner(3, 3) = q.toRotationMatrix();
    std::cout << "T=\n"
              << T << std::endl;

    zT = Eigen::Matrix4f::Identity();
    zT.topLeftCorner(3, 3) = Eigen::AngleAxisf(z_angle * 3.1415f / 180.f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    std::cout << "zT=\n"
              << zT << std::endl;
  }
  Eigen::Matrix4f T;
  Eigen::Matrix4f zT;
  Eigen::Vector3f max_box, min_box;
  Eigen::Vector3f normal;
  std::string pcd_file;
  float leaf_size;
  float z_angle;
};

int main(int argc, char** argv)
{
  if (argc == 1) {
    return -1;
  }

  bool save = false;
  std::string yaml_file = argv[1];
  if (argc == 3)
    save = true;

  Config config(yaml_file);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(config.pcd_file, *cloud) == -1) {
    std::cout << "Couldn't read file " << config.pcd_file << std::endl;
    exit(1);
  }
  std::cout << "size=" << cloud->size() << std::endl;

  if (!save) {
    // filtering
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud);
    filter.setLeafSize(config.leaf_size, config.leaf_size, config.leaf_size);
    filter.filter(*cloud);
  }

  pcl::transformPointCloud(*cloud, *cloud, config.zT);
  pcl::transformPointCloud(*cloud, *cloud, config.T);

  cloud = cropPointCloud(cloud, config.min_box, config.max_box);
  if (save) {
    pcl::io::savePCDFileBinary("output.pcd", *cloud);
    std::cout << "output.pcd" << std::endl;
    return 0;
  }

  // rgb pointcloud viewer
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  setColor(cloud, color_cloud);
  pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("viewer"));
  viewer->addPointCloud<pcl::PointXYZRGBA>(color_cloud, "sample cloud");
  viewer->addCoordinateSystem(2);

  std::string cube_name = "cube";
  Eigen::Vector3f minus = (config.max_box - config.min_box) / 2.f;
  Eigen::Vector3f plus = (config.max_box + config.min_box) / 2.f;
  viewer->addCube(plus, Eigen::Quaternionf::Identity(), 2 * minus.x(), 2 * minus.y(), 0.5, cube_name);
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, cube_name);
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, cube_name);

  viewer->spin();
}