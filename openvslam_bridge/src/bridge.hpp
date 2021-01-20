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

#pragma once
#include <opencv2/videoio.hpp>
#include <openvslam/system.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace iris
{
class BridgeOpenVSLAM
{
public:
  BridgeOpenVSLAM() {}
  ~BridgeOpenVSLAM();

  void virtual setup(const std::string& config_path, const std::string& vocab_path);
  void execute(const cv::Mat& image);
  void requestReset();

  void getLandmarksAndNormals(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& vslam_data, float height) const;
  void setCriteria(unsigned int recollection_, float accuracy_);
  std::pair<unsigned int, float> getCriteria() const;

  // return openvslam::tracker_state_t
  int getState() const;

  cv::Mat getFrame() const;

  Eigen::Matrix4f getCameraPose() const;

protected:
  unsigned int recollection = 0;
  float accuracy = -1;

  std::shared_ptr<openvslam::system> SLAM_ptr = nullptr;
};
}  // namespace iris
