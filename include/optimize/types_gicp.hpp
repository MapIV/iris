#pragma once
#include <Eigen/Geometry>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

namespace vllm
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
}  // namespace vllm
