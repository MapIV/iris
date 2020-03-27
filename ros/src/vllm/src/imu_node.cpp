#include "ekf.hpp"
#include <chrono>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

void imuCallback(const sensor_msgs::Imu&)
{
  std::cout << "It subscribed 'camera/imu'" << std::endl;
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vllm_node");

  ros::NodeHandle nh;
  ros::Subscriber sub_imu = nh.subscribe("/camera/imu", 10, &imuCallback);

  ros::spin();
}
