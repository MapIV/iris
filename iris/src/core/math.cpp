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

#include "core/math.hpp"

namespace iris
{
namespace so3
{
namespace
{
constexpr float EPSILON = 1e-7f;
}
Eigen::Matrix3f hat(const Eigen::Vector3f& xi)
{
  Eigen::Matrix3f S;
  // clang-format off
  S <<  0,   -xi(2), xi(1),
       xi(2), 0,    -xi(0),
      -xi(1), xi(0), 0;
  // clang-format on
  return S;
}

Eigen::Vector3f log(const Eigen::Matrix3f& R)
{
  Eigen::Vector3f xi = Eigen::Vector3f::Zero();
  float w_length = static_cast<float>(std::acos((R.trace() - 1.0f) * 0.5f));
  if (w_length > EPSILON) {
    Eigen::Vector3f tmp;
    tmp << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    xi = 1.0f / (2.0f * static_cast<float>(std::sin(w_length))) * tmp * w_length;
  }
  return xi;
}

Eigen::Matrix3f exp(const Eigen::Vector3f& xi)
{
  float theta = xi.norm();
  Eigen::Vector3f axis = xi.normalized();

  float cos = std::cos(theta);
  float sin = std::sin(theta);

  auto tmp1 = cos * Eigen::Matrix3f::Identity();
  auto tmp2 = (1 - cos) * (axis * axis.transpose());
  auto tmp3 = sin * hat(axis);
  return tmp1 + tmp2 + tmp3;
}
}  // namespace so3
}  // namespace iris