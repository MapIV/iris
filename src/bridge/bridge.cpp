#include "vllm/bridge/bridge.hpp"
#include "openvslam/config.h"
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

namespace vllm
{

BridgeOpenVSLAM::~BridgeOpenVSLAM()
{
  if (SLAM_ptr == nullptr)
    return;

  // wait until the loop BA is finished
  while (SLAM_ptr->loop_BA_is_running()) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
  }

  // shutdown the SLAM process
  SLAM_ptr->shutdown();
}

void BridgeOpenVSLAM::setup(const Config& config)
{
  // setup logger
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
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

unsigned int BridgeOpenVSLAM::getPeriodFromInitialId() const
{
  return SLAM_ptr->get_frame_publisher()->period_from_initial_id_;
}


void BridgeOpenVSLAM::getLandmarksAndNormals(pcXYZIN::Ptr& vslam_data) const
{
  if (recollection == 0 || accuracy < 0) {
    std::cerr << "ERROR: recollection & accuracy are not set" << std::endl;
    exit(1);
  }

  std::vector<openvslam::data::landmark*> landmarks;
  std::set<openvslam::data::landmark*> local_landmarks;
  SLAM_ptr->get_map_publisher()->get_landmarks(landmarks, local_landmarks);

  vslam_data->clear();

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
    // NOTE:OBSL: Newly observed points have priority
    // weight = static_cast<float>(recollection - (max_id - first_observed_id)) / static_cast<float>(recollection);
    if (weight < 0.1f) weight = 0.1f;
    if (weight > 1.0f) weight = 1.0f;

    pcl::PointXYZINormal p;
    p.x = static_cast<float>(pos.x());
    p.y = static_cast<float>(pos.y());
    p.z = static_cast<float>(pos.z());
    p.normal_x = static_cast<float>(normal.x());
    p.normal_y = static_cast<float>(normal.y());
    p.normal_z = static_cast<float>(normal.z());
    p.intensity = weight;
    vslam_data->push_back(p);
  }

  std::cout
      << "landmark ratio \033[34m" << vslam_data->size()
      << "\033[m / \033[34m" << local_landmarks.size()
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

Eigen::Matrix4f BridgeOpenVSLAM::getCameraPose() const
{
  return SLAM_ptr->get_map_publisher()->get_current_cam_pose().cast<float>();
}

void BridgeOpenVSLAM::requestReset()
{
  if (!SLAM_ptr->reset_is_requested())
    SLAM_ptr->request_reset();
}

void BridgeOpenVSLAM::execute(const cv::Mat& image)
{
  SLAM_ptr->feed_monocular_frame(image, 0.05, cv::Mat{});
  if (SLAM_ptr->get_frame_publisher()->get_tracking_state() == openvslam::tracker_state_t::Lost) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    requestReset();
  }
}

void BridgeOpenVSLAM::setCriteria(unsigned int recollection_, float accuracy_)
{
  recollection = recollection_;
  accuracy = accuracy_;

  if (recollection < 1) recollection = 1;
  if (accuracy < 0.1f) accuracy = 0.1f;
  if (accuracy > 1.0f) accuracy = 1.0f;
}

std::pair<unsigned int, float> BridgeOpenVSLAM::getCriteria() const
{
  return {recollection, accuracy};
}

}  // namespace vllm