#include "bridge.hpp"
#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
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

void BridgeOpenVSLAM::setup(int argc, char* argv[])
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
  auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");

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
  if (cfg->camera_->setup_type_ != openvslam::camera::setup_type_t::Monocular) {
    std::cout << "Invalid setup type: " + cfg->camera_->get_setup_type_string() << std::endl;
    exit(EXIT_FAILURE);
  }

  video = cv::VideoCapture(video_file_path->value(), cv::CAP_FFMPEG);

  // build a SLAM system
  SLAM_ptr = std::make_shared<openvslam::system>(cfg, vocab_file_path->value());
  SLAM_ptr->startup();
}

bool BridgeOpenVSLAM::execute()
{
  if (!is_not_end)
    return false;

  cv::Mat frame;
  is_not_end = video.read(frame);

  if (!frame.empty()) {
    // input the current frame and estimate the camera pose
     SLAM_ptr->feed_monocular_frame(frame, 0.05, cv::Mat{});
  } else {
    return false;
  }

  // return success
  return true;
}
