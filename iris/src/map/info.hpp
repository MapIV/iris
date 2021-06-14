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
#include <Eigen/Dense>
#include <limits>

namespace iris
{
namespace map
{
struct Info {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  float x = std::numeric_limits<float>::quiet_NaN();
  float y = std::numeric_limits<float>::quiet_NaN();
  float theta = std::numeric_limits<float>::quiet_NaN();

  Info() {}
  Info(float x, float y, float theta) : x(x), y(y), theta(theta) {}

  Eigen::Vector2f xy() const { return Eigen::Vector2f(x, y); }

  std::string toString() const
  {
    return std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(theta);
  }

  bool isEqual(const Info& a, const Info& b) const
  {
    constexpr float EPSILON = 1e-7f;
    if (std::fabs(a.x - b.x) > EPSILON)
      return false;
    if (std::fabs(a.y - b.y) > EPSILON)
      return false;
    if (std::fabs(a.theta - b.theta) > EPSILON)
      return false;
    return true;
  }

  bool operator==(const Info& other) const
  {
    return isEqual(*this, other);
  }
  bool operator!=(const Info& other) const
  {
    return !isEqual(*this, other);
  }
};

}  // namespace map
}  // namespace iris