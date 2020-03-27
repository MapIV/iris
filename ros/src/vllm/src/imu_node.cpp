#include "ekf.hpp"
#include <geometry_msgs/Transform.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

vllm::EKF ekf;

std::pair<Eigen::Vector3f, Eigen::Vector3f> decomposeIMU(const sensor_msgs::Imu& imu_msg)
{
  Eigen::Vector3f a, w;
  a(0) = imu_msg.linear_acceleration.x;
  a(1) = imu_msg.linear_acceleration.y;
  a(2) = imu_msg.linear_acceleration.z;
  w(0) = imu_msg.angular_velocity.x;  // roll
  w(1) = imu_msg.angular_velocity.y;  // pitch
  w(2) = imu_msg.angular_velocity.z;  // yaw
  return {a, w};
}

void imuCallback(const sensor_msgs::Imu& msg)
{
  std::cout << "It subscribed 'camera/imu'" << std::endl;
  auto [a, w] = decomposeIMU(msg);
  ekf.predict(a, w, 0.01);
}


int main(int argc, char* argv[])
{
  ros::init(argc, argv, "imu_node");

  ros::NodeHandle nh;
  ros::Subscriber sub_imu = nh.subscribe("/camera/imu", 10, &imuCallback);

  // tf::TransformBroadcaster br;
  // tf::Transform transform;
  // transform.setOrigin(tf::Vector3(msg->x, msg->y, 0.0));
  // tf::Quaternion q;
  // q.setRPY(0, 0, msg->theta);
  // transform.setRotation(q);
  // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "vllm"));

  ros::Publisher chatter_pub = nh.advertise<geometry_msgs::Transform>("vllm/predict", 1000);

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
