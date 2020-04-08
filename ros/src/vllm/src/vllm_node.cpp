#include "vllm/core/config.hpp"
#include "vllm/map/map.hpp"
#include "vllm/system/system.hpp"
#include "vllm/viewer/pangolin_viewer.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>
#include <popl.hpp>
#include <ros/ros.h>
#include <vllm/dynamicConfig.h>

cv::Mat subscribed_image;
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  subscribed_image = cv_ptr->image.clone();
}

void callback(vllm::dynamicConfig&, uint32_t)
{
}

class Kusa
{
private:
  mutable std::mutex mtx;

public:
  void hoge()
  {
    std::lock_guard lock(mtx);
  }
};


const std::string WINDOW_NAME = "Visual Localization in 3D LiDAR Map";

int main(int argc, char* argv[])
{
  // Initialzie ROS & subscriber
  ros::init(argc, argv, "vllm_node");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("camera/color/image_raw", 1, &imageCallback);

  Kusa kusa;

  // dynamic reconfigure // NOTE: future works
  // {
  //   dynamic_reconfigure::Server<vllm::dynamicConfig> server;
  //   dynamic_reconfigure::Server<vllm::dynamicConfig>::CallbackType f = boost::bind(&callback, _1, _2);
  //   server.setCallback(f);
  // }

  // Analyze arugments
  popl::OptionParser op("Allowed options");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!config_file_path->is_set()) {
    std::cerr << "invalid arguments" << std::endl;
    std::cout << op.help() << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize config
  vllm::Config config(config_file_path->value());

  // Load LiDAR map
  vllm::map::Parameter map_param(
      config.pcd_file, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);
  std::chrono::system_clock::time_point m_start;

  // Initialize viewer
  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // The main loop runs at this frequency at most
  ros::Rate loop_10Hz(10);

  // Setup publisher
  ros::Publisher target_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("target_pointcloud", 10);
  ros::Publisher source_pc_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("source_pointcloud", 10);
  image_transport::ImageTransport it(nh);
  image_transport::Publisher image_publisher = it.advertise("image", 10);

  // Main loop
  int loop_count = 0;
  while (ros::ok()) {
    m_start = std::chrono::system_clock::now();

    if (!subscribed_image.empty()) {
      // Execution
      system->execute(subscribed_image);
      subscribed_image = cv::Mat();

      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", system->getFrame()).toImageMsg();
      image_publisher.publish(msg);

      vllm::Publication p;
      bool a = system->popPublication(p);
    }

    // inform processing time
    {
      std::stringstream ss;
      ss << "time= \033[35m"
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
         << "\033[m ms";
      ROS_INFO("%s", ss.str().c_str());
    }

    // publish heavy data
    if (++loop_count >= 100) {
      loop_count = 0;
      auto msg = map->getTargetCloud()->makeShared();
      msg->header.frame_id = "base_link";
      pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
      target_pc_publisher.publish(msg);
    }


    // spin & wait
    ros::spinOnce();
    loop_10Hz.sleep();
  }
  ROS_INFO("Finalize the system");

  return 0;
}