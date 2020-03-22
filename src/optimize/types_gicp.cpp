#include "optimize/types_gicp.hpp"

namespace vllm
{
namespace optimize
{
EdgeGICP::EdgeGICP()
{
  pos0.setZero();
  pos1.setZero();
  normal0 << 0, 0, 1;
  normal1 << 0, 0, 1;
  R0.setIdentity();
  R1.setIdentity();
}

void EdgeGICP::makeRot0()
{
  Vector3 y;
  y << 0, 1, 0;
  R0.row(2) = normal0;
  y = y - normal0(1) * normal0;
  y.normalize();
  R0.row(1) = y;
  R0.row(0) = normal0.cross(R0.row(1));
}

void EdgeGICP::makeRot1()
{
  Vector3 y;
  y << 0, 1, 0;
  R1.row(2) = normal1;
  y = y - normal1(1) * normal1;
  y.normalize();
  R1.row(1) = y;
  R1.row(0) = normal1.cross(R1.row(1));
}

Matrix3 EdgeGICP::prec0(number_t e)
{
  makeRot0();
  Matrix3 prec;
  prec << e, 0, 0,
      0, e, 0,
      0, 0, 1;
  return R0.transpose() * prec * R0;
}

Matrix3 EdgeGICP::prec1(number_t e)
{
  makeRot1();
  Matrix3 prec;
  prec << e, 0, 0,
      0, e, 0,
      0, 0, 1;
  return R1.transpose() * prec * R1;
}

Matrix3 EdgeGICP::cov0(number_t e)
{
  makeRot0();
  Matrix3 cov;
  cov << 1, 0, 0,
      0, 1, 0,
      0, 0, e;
  return R0.transpose() * cov * R0;
}
Matrix3 EdgeGICP::cov1(number_t e)
{
  makeRot1();
  Matrix3 cov;
  cov << 1, 0, 0,
      0, 1, 0,
      0, 0, e;
  return R1.transpose() * cov * R1;
}

void Edge_Sim3_GICP::computeError()
{
  // from <ViewPoint> to <Point>
  const VertexSim3Expmap* vp0 = static_cast<const VertexSim3Expmap*>(_vertices[0]);
  // get vp1 point into vp0 frame could be more efficient if we computed this transform just once
  Vector3 p1 = vp0->estimate().map(measurement().pos1);
  // Euclidean distance
  _error = p1 - measurement().pos0;

  if (!plane2plane)
    return;

  // NOTE: re-define the information matrix for Plane2Plane ICP
  const Matrix3 R = vp0->estimate().rotation().matrix();
  information() = (cov0 + R * cov1 * R.transpose()).inverse();
}

}  // namespace optimize
}  // namespace vllm