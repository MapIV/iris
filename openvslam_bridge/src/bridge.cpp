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

#include "bridge.hpp"
#include <openvslam/config.h>
#include <openvslam/data/landmark.h>
#include <openvslam/publish/frame_publisher.h>
#include <openvslam/publish/map_publisher.h>

#include <chrono>
#include <iostream>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

namespace iris
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

void BridgeOpenVSLAM::setup(const std::string& config_path, const std::string& vocab_path)
{
  // setup logger
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  spdlog::set_level(spdlog::level::info);

  // load configuration
  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = std::make_shared<openvslam::config>(config_path);
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
  SLAM_ptr = std::make_shared<openvslam::system>(cfg, vocab_path);
  SLAM_ptr->startup();
  SLAM_ptr->disable_loop_detector();
}

void BridgeOpenVSLAM::getLandmarksAndNormals(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& vslam_data, float height) const
{
  auto t0 = std::chrono::system_clock::now();

  if (recollection == 0 || accuracy < 0) {
    std::cerr << "ERROR: recollection & accuracy are not set" << std::endl;
    exit(1);
  }

  std::set<openvslam::data::landmark*> local_landmarks;
  SLAM_ptr->get_map_publisher()->get_landmarks(local_landmarks);

  vslam_data->clear();

  if (local_landmarks.empty()) return;

  Eigen::Vector3d t_vslam = SLAM_ptr->get_map_publisher()->get_current_cam_pose().topRightCorner(3, 1);

  unsigned int max_id = SLAM_ptr->get_map_publisher()->get_max_keyframe_id();
  for (const auto local_lm : local_landmarks) {
    unsigned int first_observed_id = local_lm->first_keyfrm_id_;
    unsigned int last_observed_id = local_lm->last_observed_keyfrm_id_;
    if (local_lm->will_be_erased()) continue;
    if (local_lm->get_observed_ratio() < accuracy) continue;
    if (max_id > recollection && last_observed_id < max_id - recollection) continue;

    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    // const openvslam::Vec3_t normal = local_lm->get_obs_mean_normal();

    // when the distance is 5m or more, the weight is minimum.
    // float weight = static_cast<float>(1.0 - (t_vslam - pos).norm() * 0.2);
    float weight = 1.0f;
    weight = std::min(std::max(weight, 0.1f), 1.0f);

    Eigen::Vector3f t = getCameraPose().inverse().topRightCorner(3, 1);
    if (pos.y() - t.y() < -height) continue;

    pcl::PointXYZINormal p;
    p.x = static_cast<float>(pos.x());
    p.y = static_cast<float>(pos.y());
    p.z = static_cast<float>(pos.z());
    // p.normal_x = static_cast<float>(normal.x());
    // p.normal_y = static_cast<float>(normal.y());
    // p.normal_z = static_cast<float>(normal.z());
    p.intensity = weight;
    vslam_data->push_back(p);
  }

  auto t1 = std::chrono::system_clock::now();
  long dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout
      << "landmark ratio \033[34m" << vslam_data->size()
      << "\033[m / \033[34m" << local_landmarks.size()
      << "\033[m" << dt << std::endl;

  return;
}

int BridgeOpenVSLAM::getState() const
{
  return static_cast<int>(SLAM_ptr->get_frame_publisher()->get_tracking_state());
}

cv::Mat BridgeOpenVSLAM::getFrame() const
{
  return SLAM_ptr->get_frame_publisher()->draw_frame(false);
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

  recollection = std::max(recollection, 1u);
  accuracy = std::max(accuracy, 0.1f);
  accuracy = std::min(accuracy, 1.0f);
}

std::pair<unsigned int, float> BridgeOpenVSLAM::getCriteria() const
{
  return {recollection, accuracy};
}

}  // namespace iris
