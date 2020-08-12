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

#include "optimize/types_restriction.hpp"

namespace iris
{
namespace optimize
{
void Edge_Scale_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  double scale = vp0->estimate().scale();
  const double ref_scale = measurement();

  double diff = (ref_scale - scale);
  double e = 0;
  if (diff > 0) e = diff;
  if (diff < 0) e = -diff / (scale + 1e-6);

  _error(0) = gain * e;
}

void Edge_Altitude_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  Eigen::Vector3d now = vp0->estimate().map(measurement());
  _error(0) = gain * now.z();
}

void Edge_Latitude_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);

  // Because Visual-SLAM hundle the direction in front of camera as the Z-axis,
  // the alignment transform contains the rotation which converts Z-axis(in front of camera) to X-axis(in front of base_link)
  Eigen::Matrix3d R = vp0->estimate().rotation().toRotationMatrix() * offset_rotation;
  Eigen::Vector3d b(0, -1, 0);

  Eigen::Vector3d Rb = R * b;
  // Rb get (0,0,1) when the camera doesn't pitch and roll.
  // If the camera roll,  Rb gets approximately (0, e, 1-e).
  // If the camera pitch, Rb gets approximately (e, 0, 1-e).
  // Therefore, 1-Rb.z() means how the camera roll or pitch.

  double swing = 1 - Rb.z();

  // an angle of the camera rolling and pitching larger than acos(0.75) = 41[deg]
  if (swing > 0.20)
    _error(0) = 1e4;  // infinity loss
  // the angle is enough small.
  else
    _error(0) = gain * swing;
}

void Edge_Euclid_Restriction::computeError()
{
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);

  // Eigen::Matrix3d R = vp0->estimate().rotation().toRotationMatrix();
  Eigen::Vector3d t = vp0->estimate().translation();
  // double s = vp0->estimate().scale();

  // double e1 = (R - R_init).trace();
  double e2 = (t - t_init).norm();
  // double e3 = (s - s_init);

  // TODO: NOTE:
  _error(0) = gain * e2;
}

// void Edge_Smooth_Restriction::computeError()
// {
// TODO:
// const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);

// double min=
// Eigen::Vector3f t0 = measurement().at(0);
// for (size_t i = 1, N = measurement().size(); i < N - 1; i++) {
//   Eigen::Vector3f t1 = measurement().at(i);

//   t0 = t1;
// }
// Eigen::Vector3d older = measurement().older_pos;
// Eigen::Vector3d old = measurement().old_pos;
// Eigen::Vector3d now = vp0->estimate().map(measurement().camera_pos);

// double old_vel = (old - older).norm();
// double vel = (now - old).norm();

// double thr = 0.1;
// double dv = vel - old_vel;
// // Usually, the larger the velocity, the better
// if (dv < thr)
//   _error(0) = gain * (thr - dv) + 0.5 * (vel - old_vel);
// else
//   _error(0) = 0.5 * (vel - old_vel);
// }

}  // namespace optimize
}  // namespace iris