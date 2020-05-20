#include "bridge.hpp"
#include <iostream>
#include <ros/ros.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "openvslam_bridge_node");

  ros::Time::init();
  ros::Rate loop_rate(5);

  vllm::BridgeOpenVSLAM bridge;

  while (ros::ok()) {
    ROS_INFO("openvslam_bridge_node");
    loop_rate.sleep();
    ros::spinOnce();
  }
}