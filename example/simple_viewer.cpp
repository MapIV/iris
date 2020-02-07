#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using pcXYZ = pcl::PointCloud<pcl::PointXYZ>;
using pcXYZI = pcl::PointCloud<pcl::PointXYZI>;

pcXYZI::Ptr cropPointCloud(pcXYZI::Ptr source, Eigen::Vector3f min, Eigen::Vector3f max)
{
  pcXYZI::Ptr cropped(new pcXYZI);
  pcl::CropBox<pcl::PointXYZI> crop;

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

void transformPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
  // rotation
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(static_cast<float>(-M_PI / 30.0), Eigen::Vector3f(0, 0, 1));

  // transration
  Eigen::Vector3f t;
  t << -1.5f, 2.1f, -0.1f;

  // transform
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = R;
  T.topRightCorner(3, 1) = t;

  pcl::transformPointCloud(*cloud, *cloud, T);
}

void setColor(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& gray,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color)
{
  for (size_t i = 0, N = gray->size(); i < N; i++) {
    pcl::PointXYZRGB p;
    p.x = gray->at(i).x;
    p.y = gray->at(i).y;
    p.z = gray->at(i).z;
    p.r = std::abs(p.x) * 255.f / 1.0;
    p.g = std::abs(p.y) * 255.f / 1.0;
    p.b = std::abs(p.z) * 255.f / 1.0;
    color->push_back(p);
  }
}


int main(int argc, char** argv)
{
  if (argc != 2) {
    return -1;
  }

  std::string pcd_file = argv[1];
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
    std::cout << "Couldn't read file " << pcd_file << std::endl;
    exit(1);
  }

  transformPointCloud(cloud);
  cloud = cropPointCloud(cloud, Eigen::Vector3f(-2, -1, -1), Eigen::Vector3f(1, 5, 2));

  // // intensity viewer
  // pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("viewer"));
  // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "intensity");
  // viewer->addPointCloud<pcl::PointXYZI>(cloud, color_handler, "sample cloud");
  // viewer->addCoordinateSystem(0.5);

  // rgb pointcloud viewer
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  setColor(cloud, color_cloud);
  pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("viewer"));
  viewer->addPointCloud<pcl::PointXYZRGB>(color_cloud, "sample cloud");
  viewer->addCoordinateSystem(0.5);


  // std::string cube_name = "cube";
  // viewer->addCube(Eigen::Vector3f::Zero(), Eigen::Quaternionf(), 3, 6, 3, cube_name);
  // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, cube_name);
  // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, cube_name);
  // pcl::io::savePCDFileBinary("room.pcd", *cloud);

  viewer->spin();
}