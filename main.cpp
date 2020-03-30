#include "core/config.hpp"
#include "imu/ekf.hpp"
#include "imu/topic.hpp"
#include "map/map.hpp"
#include "system/system.hpp"
#include "viewer/pangolin_viewer.hpp"
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <popl.hpp>

int main(int argc, char* argv[])
{
  // analyze arugments
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
    exit(EXIT_FAILURE);
  }

  // Initialize config
  vllm::Config config(config_file_path->value());

  // Load LiDAR map
  vllm::map::Parameter map_param(
      config.pcd_file, config.voxel_grid_leaf, config.normal_search_leaf, config.submap_grid_leaf);
  std::shared_ptr<vllm::map::Map> map = std::make_shared<vllm::map::Map>(map_param);

  // Initialize system
  // std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);

  // Initialize viewer
  // vllm::viewer::PangolinViewer pangolin_viewer(system);
  // pangolin_viewer.startLoop();
  // cv::namedWindow("VLLM", cv::WINDOW_AUTOSIZE);

  // video
  cv::VideoCapture video = cv::VideoCapture(config.video_file, cv::CAP_FFMPEG);

  // for IMU
  vllm::TopicAnalyzer topic(config.topic_file, config.imu_file);
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity(4, 4);
  Eigen::Matrix3f R;
  R << 1, 0, 0,
      0, -1, 0,
      0, 0, -1;
  T.topLeftCorner(3, 3) = R;
  std::cout << T << std::endl;
  vllm::EKF ekf(T);

  std::chrono::system_clock::time_point m_start;
  unsigned int time = 0;

  while (true) {
    bool is_topic_video = topic.isTopicVideo(time);

    m_start = std::chrono::system_clock::now();

    if (is_topic_video) {
      std::cout << "video" << std::endl;
      // // Read frame from video
      // cv::Mat frame;
      // bool is_not_end = true;
      // for (int i = 0; i < config.frame_skip && is_not_end; i++) is_not_end = video.read(frame);
      // if (!is_not_end) break;

      // // Execution
      // system->execute(frame);
    } else {
      Eigen::Vector3f acc = topic.getAcc(time);
      Eigen::Vector3f omega = topic.getOmega(time);

      ekf.predict(acc, omega, 0.002);
      Eigen::Matrix4f T = ekf.getState();
    }

    std::cout << "time= \033[35m"
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
              << "\033[m ms" << std::endl;

    // visualize by OpenCV
    //cv::imshow("VLLM", system->getFrame());
    int key = cv::waitKey(1);
    if (key == 'q') break;
    if (key == 's') {
      while (key == 's')
        key = cv::waitKey(0);
    }

    time++;
  }

  // pangolin_viewer.quitLoop();
  return 0;
}
