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
#include "core/util.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

namespace iris
{
namespace optimize
{
using g2o::VertexSim3Expmap;

class VelocityModel
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d camera_pos, old_pos, older_pos;

  VelocityModel()
  {
    old_pos.setZero();
    older_pos.setZero();
    camera_pos.setZero();
  }

  double velocity() const
  {
    return (old_pos - older_pos).norm();
  }
};

class Edge_Scale_Restriction : public g2o::BaseUnaryEdge<1, double, VertexSim3Expmap>
{
private:
  double gain;

public:
  Edge_Scale_Restriction(double gain = 1.0) : gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

class Edge_Altitude_Restriction : public g2o::BaseUnaryEdge<1, Eigen::Vector3d, VertexSim3Expmap>
{
private:
  double gain;

public:
  Edge_Altitude_Restriction(double gain = 1.0) : gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

class Edge_Latitude_Restriction : public g2o::BaseUnaryEdge<1, double, VertexSim3Expmap>
{
private:
  Eigen::Matrix3d offset_rotation;
  double gain;

public:
  Edge_Latitude_Restriction(const Eigen::Matrix3d& offset_rotation, double gain = 1.0) : offset_rotation(offset_rotation),
                                                                                         gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

class Edge_Euclid_Restriction : public g2o::BaseUnaryEdge<1, double, VertexSim3Expmap>
{
private:
  Eigen::Matrix3d R_init;
  Eigen::Vector3d t_init;
  double s_init;
  double gain;

public:
  Edge_Euclid_Restriction(const Eigen::Matrix4f& T_init, double gain = 1.0) : gain(gain)
  {
    Eigen::Matrix3f sR = T_init.topLeftCorner(3, 3);
    R_init = util::normalizeRotation(sR).cast<double>();
    t_init = T_init.topRightCorner(3, 1).cast<double>();
    s_init = util::getScale(sR);
  }

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

// class Edge_Smooth_Restriction : public g2o::BaseUnaryEdge<1, std::vector<Eigen::Vector3f>, VertexSim3Expmap>
// {
// private:
//   double gain;

// public:
//   Edge_Smooth_Restriction(double gain = 1.0) : gain(gain) {}

//   virtual bool read(std::istream&) { return false; }
//   virtual bool write(std::ostream&) const { return false; }
//   void computeError();
// };


}  // namespace optimize
}  // namespace iris
