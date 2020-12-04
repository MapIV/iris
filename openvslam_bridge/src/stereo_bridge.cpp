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

void BridgeStereoOpenVSLAM::setup(const std::string& config_path, const std::string& vocab_path)
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
  if (cfg->camera_->setup_type_ != openvslam::camera::setup_type_t::Stereo) {
    std::cout << "Invalid setup type: " + cfg->camera_->get_setup_type_string() << std::endl;
    exit(EXIT_FAILURE);
  }

  // build a SLAM system
  SLAM_ptr = std::make_shared<openvslam::system>(cfg, vocab_path);
  SLAM_ptr->startup();
  SLAM_ptr->disable_loop_detector();
}

void BridgeStereoOpenVSLAM::execute(const cv::Mat& image0, const cv::Mat& image1)
{
  SLAM_ptr->feed_stereo_frame(image0, image1, 0.05, cv::Mat{});

  if (SLAM_ptr->get_frame_publisher()->get_tracking_state() == openvslam::tracker_state_t::Lost) {
    std::cout << "\n\033[33m ##### Request Reset #####\n\033[m" << std::endl;
    requestReset();
  }
}


}  // namespace iris