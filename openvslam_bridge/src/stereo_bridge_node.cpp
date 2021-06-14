// Copyright (c) 2020, Map IV, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the Map IV, Inc. nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "stereo_bridge.hpp"
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <iostream>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
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
  ros::init(argc, argv, "openvslam_stereo_bridge_node");

  // Get rosparams
  ros::NodeHandle pnh("~");
  bool is_image_compressed = true;
  bool is_image_color = true;
  int recollection = 30;
  float height = std::numeric_limits<float>::max();
  std::string vocab_path, vslam_config_path;
  std::string image_topic_name0;
  std::string image_topic_name1;
  pnh.getParam("vocab_path", vocab_path);
  pnh.getParam("vslam_config_path", vslam_config_path);
  pnh.getParam("image_topic_name0", image_topic_name0);
  pnh.getParam("image_topic_name1", image_topic_name1);
  pnh.getParam("is_image_compressed", is_image_compressed);
  pnh.getParam("is_image_color", is_image_color);
  pnh.getParam("keyframe_recollection", recollection);
  pnh.getParam("max_height", height);
  ROS_INFO("vocab_path: %s, vslam_config_path: %s, image_topic_name: %s, is_image_compressed: %d",
      vocab_path.c_str(), vslam_config_path.c_str(), image_topic_name0.c_str(), is_image_compressed);

  const int lower_threshold_of_points = 1000;
  const int upper_threshold_of_points = 1500;

  // Setup subscriber
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  // Setup image subscriber
  message_filters::Subscriber<sensor_msgs::CompressedImage> infra1_image_subscriber(nh, image_topic_name0, 1);
  message_filters::Subscriber<sensor_msgs::CompressedImage> infra2_image_subscriber(nh, image_topic_name1, 1);
  message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> syncronizer(infra1_image_subscriber, infra2_image_subscriber, 10);


  // Setup image callback
  cv::Mat subscribed_image0, subscribed_image1;
  ros::Time subscribed_stamp;
  if (is_image_compressed) {
    auto image_callback = [is_image_color, &subscribed_image0, &subscribed_image1, &subscribed_stamp](const sensor_msgs::CompressedImageConstPtr& image0, const sensor_msgs::CompressedImageConstPtr& image1) -> void {
      subscribed_image0 = cv::imdecode(cv::Mat(image0->data), is_image_color ? 1 : 0 /* '1': bgr, '0': gray*/);
      subscribed_image1 = cv::imdecode(cv::Mat(image1->data), is_image_color ? 1 : 0 /* '1': bgr, '0': gray*/);
      subscribed_stamp = image0->header.stamp;
    };
    syncronizer.registerCallback(boost::bind<void>(image_callback, _1, _2));
  } else {
    std::cerr << "Error: Only compressed image is acceptable" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Setup publisher
  ros::Publisher vslam_publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZINormal>>("iris/vslam_data", 1);
  image_transport::Publisher image_publisher = it.advertise("iris/processed_image", 5);

  // Setup for OpenVSLAM
  iris::BridgeStereoOpenVSLAM bridge;
  bridge.setup(vslam_config_path, vocab_path);

  std::chrono::system_clock::time_point m_start;
  ros::Rate loop_rate(20);
  float accuracy = 0.5f;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr vslam_data(new pcl::PointCloud<pcl::PointXYZINormal>);
  std::ofstream ofs_time("vslam_time.csv");

  // Start main loop
  ROS_INFO("start main loop.");
  while (ros::ok()) {
    if (!subscribed_image0.empty() && !subscribed_image1.empty()) {
      // start timer
      m_start = std::chrono::system_clock::now();
      ros::Time process_stamp = subscribed_stamp;

      // process OpenVSLAM
      bridge.execute(subscribed_image0, subscribed_image1);
      bridge.setCriteria(recollection, accuracy);
      bridge.getLandmarksAndNormals(vslam_data, height);

      // Inform processing time
      std::stringstream ss;
      long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count();
      ss << "processing time= \033[35m"
         << time_ms
         << "\033[m ms";
      ofs_time << time_ms << std::endl;
      ROS_INFO("%s", ss.str().c_str());

      // Reset input
      subscribed_image0 = cv::Mat();
      subscribed_image1 = cv::Mat();

      // Update threshold to adjust the number of points
      if (vslam_data->size() < lower_threshold_of_points && accuracy > 0.10) accuracy -= 0.01f;
      if (vslam_data->size() > upper_threshold_of_points && accuracy < 0.90) accuracy += 0.01f;
      std::cout << "accuracy: " << accuracy << std::endl;
      {
        cv::Mat img = bridge.getFrame();
        if (!img.empty()) {
          sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
          image_publisher.publish(msg);
        }
      }
      {
        pcl_conversions::toPCL(process_stamp, vslam_data->header.stamp);
        vslam_data->header.frame_id = "world";
        vslam_publisher.publish(vslam_data);
      }
    }
    publishPose(bridge.getCameraPose().inverse(), "iris/vslam_pose");

    // Spin and wait
    loop_rate.sleep();
    ros::spinOnce();
  }

  ROS_INFO("Finalize openvslam_bridge::bridge_node");
  return 0;
}
