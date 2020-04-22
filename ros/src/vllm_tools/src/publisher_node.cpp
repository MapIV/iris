#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <ros/ros.h>

int main(int argc, char* argv[])
{
  // Initialzie ROS & subscriber
  ros::init(argc, argv, "publisher_node");

  // Analyze arugments
  popl::OptionParser op("Allowed options");
  auto video_file_path = op.add<popl::Value<std::string>>("v", "video", "video file path");
  auto publish_fps = op.add<popl::Value<int>>("f", "fps", "publish fps");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!video_file_path->is_set() || !publish_fps->is_set()) {
    std::cout << op.help() << std::endl;
    exit(EXIT_FAILURE);
  }

  // Setup publisher
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher image_publisher = it.advertise("/camera/color/image_raw", 10);
  cv::namedWindow("show", cv::WINDOW_NORMAL);

  // Setup loop rate
  int fps = publish_fps->value();
  fps = std::min(100, fps);
  fps = std::max(1, fps);
  ros::Rate loop_rate(fps);

  cv::VideoCapture video(video_file_path->value());
  // cv::VideoCapture video(0);
  if (!video.isOpened()) {
    std::cout << "can't open " << video_file_path->value() << std::endl;
    exit(EXIT_FAILURE);
  }

  while (ros::ok()) {
    cv::Mat frame;
    bool is_not_end = video.read(frame);
    if (!is_not_end)
      break;

    // publish
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    image_publisher.publish(msg);

    cv::imshow("show", frame);
    cv::waitKey(1);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
