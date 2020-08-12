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
#include "core/math.hpp"
#include "core/util.hpp"

namespace iris
{
namespace optimize
{
Eigen::Vector3f calcAverageTransform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t, int n)
{
  Eigen::Matrix3f A = Eigen::Matrix3f::Zero();

  for (int i = 0; i < n; i++) {
    Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
    for (int j = 0; j < i; j++) {
      tmp = R * tmp;
    }
    A += tmp;
  }

  return A.inverse() * t;
}

Eigen::Matrix4f calcVelocity(const std::list<Eigen::Matrix4f>& poses)
{
  Eigen::Matrix4f T0 = util::normalizePose(*std::next(poses.begin()));  // The latest pose seems to be not reliable
  Eigen::Matrix4f Tn = util::normalizePose(*std::prev(poses.end()));
  const int dt = static_cast<int>(poses.size()) - 2;

  Eigen::Matrix4f tmp = T0 * Tn.inverse();
  Eigen::Matrix3f R = tmp.topLeftCorner(3, 3);
  Eigen::Vector3f t = tmp.topRightCorner(3, 1);
  Eigen::Matrix3f root_R = so3::exp(so3::log(R) / dt);

  Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
  V.topLeftCorner(3, 3) = root_R;
  V.topRightCorner(3, 1) = calcAverageTransform(root_R, t, dt);
  return V;
}
}  // namespace optimize
}  // namespace iris
