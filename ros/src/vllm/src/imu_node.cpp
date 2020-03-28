#include "ekf.hpp"
#include <geometry_msgs/Transform.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

vllm::EKF ekf;

void imuCallback(const sensor_msgs::Imu& msg)
{
  Eigen::Vector3f a, w;
  a(0) = msg.linear_acceleration.x;
  a(1) = msg.linear_acceleration.y;
  a(2) = msg.linear_acceleration.z;
  w(0) = msg.angular_velocity.x;  // roll
  w(1) = msg.angular_velocity.y;  // pitch
  w(2) = msg.angular_velocity.z;  // yaw

  std::cout << "ekf::predict" << std::endl;
  ekf.predict(a, w, 0.01);
}

void vllmCallback(const geometry_msgs::Transform& msg)
{
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "imu_node");

  ros::NodeHandle nh;
  ros::Subscriber sub_imu = nh.subscribe("/camera/imu", 10, &imuCallback);

  ros::Publisher pub_predict = nh.advertise<geometry_msgs::Transform>("vllm/predict", 1000);

  ros::Rate loop_rate(10);
  while (true) {
    geometry_msgs::Transform msg;

    msg.translation.x = 0;
    msg.translation.y = 1;
    msg.translation.z = 2;
    msg.rotation.w = 1;
    msg.rotation.x = 0;
    msg.rotation.y = 0;
    msg.rotation.z = 0;

    chatter_pub.publish(msg);


    ros::spinOnce();

    loop_rate.sleep();
  }
}
