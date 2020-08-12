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
using g2o::Matrix3;
using g2o::Vector3;
using g2o::VertexSim3Expmap;

class EdgeGICP
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // point positions
  Vector3 pos0, pos1;
  // unit normals
  Vector3 normal0, normal1;
  // rotation matrix for normal
  Matrix3 R0, R1;
  double weight;
  bool plane2plane;

  EdgeGICP();

  // set up rotation matrix for pos0, pos1
  void makeRot0();  // for target
  void makeRot1();  // for source

  // returns a precision matrix for point-plane
  Matrix3 prec0(number_t e);  // for target
  Matrix3 prec1(number_t e);  // for source

  // return a covariance matrix for plane-plane
  Matrix3 cov0(number_t e);  // for target
  Matrix3 cov1(number_t e);  // for source
};

class Edge_Sim3_GICP : public g2o::BaseUnaryEdge<3, EdgeGICP, VertexSim3Expmap>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge_Sim3_GICP(bool pl_pl = false) : plane2plane(pl_pl) {}
  Matrix3 cov0, cov1;
  bool plane2plane;

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};
}  // namespace optimize
}  // namespace iris
