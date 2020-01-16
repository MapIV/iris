#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char* argv[])
{
  // specify input file
  std::string pcd_file = "../data/table.pcd";
  if (argc == 2)
    pcd_file = argv[1];

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

  // load point cloud
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud_in) == -1) {
    std::cout << "Couldn't read file test_pcd.pcd " << pcd_file << std::endl;
    return -1;
  }
  std::cout << "Loaded "
            << cloud_in->width * cloud_in->height
            << std::endl;

  // deep copy
  *cloud_out = *cloud_in;

  // transform
  for (std::size_t i = 0; i < cloud_in->points.size(); ++i)
    cloud_out->points[i].x = cloud_in->points[i].x + 0.7f;

  // setup ICP
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  pcl::PointCloud<pcl::PointXYZ> final;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
  icp.align(final);

  //
  std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("visualizer"));
  pcl::PointCloud<pcl::PointXYZ>::Ptr final_ptr = final.makeShared();

  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(final_ptr, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(final_ptr, single_color, "cloud_final");
  }
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_in, 255, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_in, single_color, "cloud_in");
  }

  viewer->addCoordinateSystem(1.0);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
  return 0;
}