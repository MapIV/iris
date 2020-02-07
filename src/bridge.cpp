#include "bridge.hpp"
#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include <chrono>
#include <iostream>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <popl.hpp>
#include <spdlog/spdlog.h>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

void BridgeOpenVSLAM::start(int argc, char* argv[])
{
#ifdef USE_STACK_TRACE_LOGGER
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
#endif

  // create options
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto video_file_path = op.add<popl::Value<std::string>>("m", "video", "video file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
  auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
  auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");

  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    exit(EXIT_FAILURE);
  }

  // check validness of options
  if (help->is_set()) {
    std::cerr << op << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!vocab_file_path->is_set() || !video_file_path->is_set() || !config_file_path->is_set()) {
    std::cerr << "invalid arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup logger
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  if (debug_mode->is_set()) {
    spdlog::set_level(spdlog::level::debug);
  } else {
    spdlog::set_level(spdlog::level::info);
  }

  // load configuration
  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = std::make_shared<openvslam::config>(config_file_path->value());
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  // run tracking
  if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
    mono_tracking(cfg,
        vocab_file_path->value(), video_file_path->value(),
        frame_skip->value(), no_sleep->is_set(), auto_term->is_set());
  } else {
    throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
  }
}

void BridgeOpenVSLAM::mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
    const std::string& vocab_file_path, const std::string& video_file_path,
    const unsigned int frame_skip, const bool no_sleep, const bool auto_term)
{
  // build a SLAM system
  SLAM_ptr = std::make_shared<openvslam::system>(cfg, vocab_file_path);
  SLAM_ptr->startup();
  const cv::Mat MASK = cv::Mat{};

  // create a viewer object
  // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
  pangolin_viewer::viewer viewer(cfg, &*SLAM_ptr,
      SLAM_ptr->get_frame_publisher(), SLAM_ptr->get_map_publisher());
#endif

  auto video = cv::VideoCapture(video_file_path, cv::CAP_FFMPEG);
  std::vector<double> track_times;

  cv::Mat frame;
  double timestamp = 0.0;
  unsigned int num_frame = 0;
  bool is_not_end = true;
  // run the SLAM in another thread
  std::thread thread([&]() {
    while (is_not_end) {
      is_not_end = video.read(frame);

      const auto tp_1 = std::chrono::steady_clock::now();

      if (!frame.empty() && (num_frame % frame_skip == 0)) {
        // input the current frame and estimate the camera pose
        SLAM_ptr->feed_monocular_frame(frame, timestamp, MASK);
      }

      const auto tp_2 = std::chrono::steady_clock::now();

      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      if (num_frame % frame_skip == 0) {
        track_times.push_back(track_time);
      }

      // wait until the timestamp of the next frame
      if (!no_sleep) {
        const auto wait_time = 1.0 / cfg->camera_->fps_ - track_time;
        if (0.0 < wait_time) {
          std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
        }
      }

      timestamp += 1.0 / cfg->camera_->fps_;
      ++num_frame;

      // check if the termination of SLAM system is requested or not
      if (SLAM_ptr->terminate_is_requested()) {
        break;
      }
    }

    // wait until the loop BA is finished
    while (SLAM_ptr->loop_BA_is_running()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    if (auto_term) {
      viewer.request_terminate();
    }
#elif USE_SOCKET_PUBLISHER
    if (auto_term) {
      publisher.request_terminate();
    }
#endif
  });

  // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
  viewer.run();
#endif

  thread.join();

  // shutdown the SLAM process
  SLAM_ptr->shutdown();
}
