#include "bridge.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

void showPointCloud()
{
  std::string pcd_file = "../data/room.pcd";
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
    std::cout << "Couldn't read file " << pcd_file << std::endl;
    exit(1);
  }

  // rgb pointcloud viewer
  pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("viewer"));
  viewer->addPointCloud<pcl::PointXYZI>(cloud, "sample cloud");
  viewer->addCoordinateSystem(0.5);
  viewer->spin();
}

int main(int argc, char* argv[])
{
  BridgeOpenVSLAM bridge;
  std::thread vslam_thread = std::thread(&BridgeOpenVSLAM::start, bridge, argc, argv);

  auto frame_publisher = bridge.get_frame_publisher();
  auto map_publisher = bridge.get_map_publisher();

  showPointCloud();

  vslam_thread.join();
  return 0;
}
