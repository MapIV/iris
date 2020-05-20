#include "bridge.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <pcl/correspondence.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

std::function<void(const sensor_msgs::ImageConstPtr&)> imageCallbackGenerator(cv::Mat& subscribed_image)
{
  return [&subscribed_image](const sensor_msgs::ImageConstPtr& msg) -> void {
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    subscribed_image = cv_ptr->image.clone();
  };
}


void publishPose(const Eigen::Matrix4f& T, const std::string& child_frame_id)
{
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setFromOpenGLMatrix(T.cast<double>().eval().data());
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", child_frame_id));
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "openvslam_bridge_node");

  // TODO:
  // We must set the values of the following parameters using rosparam
  // - bool: is_image_comspressed
  // - string: config_path
  // - string: vocab_path
  // - int: recollection
  // - int: upper_threshold_of_pointcloud
  // - int: lower_threshold_of_pointcloud
  // - string: image_topic_name

  // Setup subscriber
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  cv::Mat subscribed_image;
  image_transport::TransportHints hints("raw");
  if (true /* is_compsressed*/) hints = image_transport::TransportHints("compressed");
  auto callback = imageCallbackGenerator(subscribed_image);
  image_transport::Subscriber image_subscriber = it.subscribe("camera/color/image_raw", 5, callback, ros::VoidPtr(), hints);

  // Setup publisher
  ros::Publisher vslam_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("vllm/vslam_data", 1);
  image_transport::Publisher image_publisher = it.advertise("vllm/processed_image", 5);

  // Setup for OpenVSLAM
  vllm::BridgeOpenVSLAM bridge;
  bridge.setup("src/vllm/config/openvslam.yaml", "src/vllm/config/orb_vocab.dbow2");

  std::chrono::system_clock::time_point m_start;
  ros::Rate loop_rate(10);
  float accuracy = 0.5f;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr vslam_data(new pcl::PointCloud<pcl::PointXYZINormal>);

  while (ros::ok()) {
    if (!subscribed_image.empty()) {
      m_start = std::chrono::system_clock::now();  // start timer

      // process OpenVSLAM
      bridge.execute(subscribed_image);
      // bridge.setCriteria(config.recollection, accuracy);
      // bridge.getLandmarksAndNormals(vslam_data);

      // Reset input
      subscribed_image = cv::Mat();

      // TODO: The accuracy should be reflected in the results of align_node
      // Update threshold to adjust the number of points
      if (vslam_data->size() < 300 && accuracy > 0.10) accuracy -= 0.01f;
      if (vslam_data->size() > 500 && accuracy < 0.90) accuracy += 0.01f;

      {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", bridge.getFrame()).toImageMsg();
        image_publisher.publish(msg);
      }
      {
        pcl_conversions::toPCL(ros::Time::now(), vslam_data->header.stamp);
        vslam_data->header.frame_id = "world";
        vslam_publisher.publish(vslam_data);
      }

      // Inform processing time
      std::stringstream ss;
      ss << "processing time= \033[35m"
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
         << "\033[m ms";
      ROS_INFO("VLLM/VSLAM: %s", ss.str().c_str());
    }
    publishPose(bridge.getCameraPose().inverse(), "vllm/vslam_pose");

    // Spin and wait
    loop_rate.sleep();
    ros::spinOnce();
  }

  ROS_INFO("Finalize openvslam_bridge::bridge_node");
  return 0;
}