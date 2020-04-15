#include "vllm/core/config.hpp"
// #include "vllm/imu/ekf.hpp"
// #include "vllm/imu/topic.hpp"
#include "vllm/map/map.hpp"
#include "vllm/system/system.hpp"
#include "vllm/viewer/pangolin_viewer.hpp"
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <popl.hpp>

const std::string WIN_NAME = "☣☣ Visual Localization in 3D LiDAR Map ☣☣";

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
  std::shared_ptr<vllm::System> system = std::make_shared<vllm::System>(config, map);

  // Initialize viewer
  vllm::viewer::PangolinViewer pangolin_viewer(system);
  pangolin_viewer.startLoop();
  cv::namedWindow(WIN_NAME, cv::WINDOW_AUTOSIZE);

  // video
  cv::VideoCapture video = cv::VideoCapture(config.video_file, cv::CAP_FFMPEG);

  // for IMU
  // vllm::TopicAnalyzer topic(config.topic_file, config.imu_file);
  // vllm::EKF ekf(config.param, config.T_init);

  std::chrono::system_clock::time_point m_start;
  unsigned int time = 0;
  int skipped_frame = 0;

  std::vector<Eigen::Matrix4f> imu_trajectory;

  while (true) {
    bool is_topic_video = true;

    if (is_topic_video) {
      // Read frame from video
      cv::Mat frame;
      if (!video.read(frame)) break;
      skipped_frame++;
      if (skipped_frame == config.frame_skip) {
        skipped_frame = 0;

        // start timer
        m_start = std::chrono::system_clock::now();

        int state = system->execute(frame);

        // stop timer
        std::cout << "time= \033[35m"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start).count()
                  << "\033[m ms" << std::endl;

        // visualize by OpenCV
        cv::imshow(WIN_NAME, system->getFrame());  // TODO: critical section
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 's') {
          while (key == 's') key = cv::waitKey(0);
        }
      } else {
        // std::cout << "image skip" << std::endl;
      }

    } else {
    }

    time++;
  }

  // pangolin_viewer.quitLoop();
  return 0;
}
