#include "bridge.hpp"
#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif
#include "openvslam/data/landmark.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"

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

void BridgeOpenVSLAM::setup(int argc, char* argv[], const std::string& video_file_path)
{
#ifdef USE_STACK_TRACE_LOGGER
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
#endif

  // create options
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  auto _frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
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

  frame_skip = _frame_skip->value();

  // check validness of options
  if (help->is_set()) {
    std::cerr << op << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!vocab_file_path->is_set() || !config_file_path->is_set()) {
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

  video = cv::VideoCapture(video_file_path, cv::CAP_FFMPEG);

  // build a SLAM system
  SLAM_ptr = std::make_shared<openvslam::system>(cfg, vocab_file_path->value());
  SLAM_ptr->startup();
}

void BridgeOpenVSLAM::getLandmarks(
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const
{
  std::vector<openvslam::data::landmark*> landmarks;
  std::set<openvslam::data::landmark*> local_landmarks;
  SLAM_ptr->get_map_publisher()->get_landmarks(landmarks, local_landmarks);

  if (landmarks.empty()) return;

  for (const auto lm : landmarks) {
    if (!lm || lm->will_be_erased()) {
      continue;
    }
    if (local_landmarks.count(lm)) {
      continue;
    }
    const openvslam::Vec3_t pos = lm->get_pos_in_world();
    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));

    cloud->push_back(p);
  }

  for (const auto local_lm : local_landmarks) {
    if (local_lm->will_be_erased()) {
      continue;
    }
    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));
    local_cloud->push_back(p);
  }
  return;
}

int BridgeOpenVSLAM::getState() const
{
  return static_cast<int>(SLAM_ptr->get_frame_publisher()->get_tracking_state());
}

cv::Mat BridgeOpenVSLAM::getFrame() const
{
  return SLAM_ptr->get_frame_publisher()->draw_frame();
}

Eigen::Matrix4d BridgeOpenVSLAM::getCameraPose() const
{
  return SLAM_ptr->get_map_publisher()->get_current_cam_pose();
}


bool BridgeOpenVSLAM::execute()
{
  if (!is_not_end)
    return false;

  cv::Mat frame;
  for (int i = 0; i < frame_skip && is_not_end; i++)
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
