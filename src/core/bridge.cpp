#include "core/bridge.hpp"
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

namespace vllm
{

BridgeOpenVSLAM::~BridgeOpenVSLAM()
{
  // wait until the loop BA is finished
  while (SLAM_ptr->loop_BA_is_running()) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
  }

  // shutdown the SLAM process
  SLAM_ptr->shutdown();
}

void BridgeOpenVSLAM::setup(const Config& config)
{
#ifdef USE_STACK_TRACE_LOGGER
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
#endif


  // setup logger
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  // spdlog::set_level(spdlog::level::debug);
  spdlog::set_level(spdlog::level::info);

  // load configuration
  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = std::make_shared<openvslam::config>(config.self_path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  // run tracking
  if (cfg->camera_->setup_type_ != openvslam::camera::setup_type_t::Monocular) {
    std::cout << "Invalid setup type: " + cfg->camera_->get_setup_type_string() << std::endl;
    exit(EXIT_FAILURE);
  }

  // build a SLAM system
  SLAM_ptr = std::make_shared<openvslam::system>(cfg, config.vocab_file);
  SLAM_ptr->startup();
  SLAM_ptr->disable_loop_detector();
}

void BridgeOpenVSLAM::getLandmarks(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const
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

unsigned int BridgeOpenVSLAM::getPeriodFromInitialId()
{
  return SLAM_ptr->get_frame_publisher()->period_from_initial_id_;
}


void BridgeOpenVSLAM::getLandmarksAndNormals(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    std::vector<float>& weights,
    unsigned int recollection,
    double accuracy) const
{
  std::vector<openvslam::data::landmark*> landmarks;
  std::set<openvslam::data::landmark*> local_landmarks;
  SLAM_ptr->get_map_publisher()->get_landmarks(landmarks, local_landmarks);

  local_cloud->clear();
  normals->clear();

  if (local_landmarks.empty()) return;

  unsigned int max_id = SLAM_ptr->get_map_publisher()->get_max_keyframe_id();
  for (const auto local_lm : landmarks) {
    unsigned int first_observed_id = local_lm->first_keyfrm_id_;
    unsigned int last_observed_id = local_lm->last_observed_keyfrm_id_;
    if (local_lm->will_be_erased()) continue;
    if (local_lm->get_observed_ratio() < accuracy) continue;
    if (max_id > recollection && last_observed_id < max_id - recollection) continue;

    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    const openvslam::Vec3_t normal = local_lm->get_obs_mean_normal();

    float weight = 1.0;
    // weight = static_cast<float>(recollection - (max_id - first_observed_id)) / static_cast<float>(recollection);
    if (weight < 0.1f) weight = 0.1f;
    if (weight > 1.0f) weight = 1.0f;
    weights.push_back(weight);

    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));
    local_cloud->push_back(p);

    pcl::Normal n(
        static_cast<float>(normal.x()),
        static_cast<float>(normal.y()),
        static_cast<float>(normal.z()));
    normals->push_back(n);
  }
  std::cout
      << "landmark ratio \033[34m" << local_cloud->size()
      << "\033[m / \033[34m" << local_landmarks.size()
      << "\033[m , latest keyframe \033[34m" << max_id
      << "\033[m , recollection \033[34m" << recollection
      << "\033[m , accuracy  \033[34m" << accuracy
      << "\033[m" << std::endl;

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

void BridgeOpenVSLAM::requestReset()
{
  if (!SLAM_ptr->reset_is_requested())
    SLAM_ptr->request_reset();
}

void BridgeOpenVSLAM::execute(const cv::Mat& image)
{
  SLAM_ptr->feed_monocular_frame(image, 0.05, cv::Mat{});
}
}  // namespace vllm