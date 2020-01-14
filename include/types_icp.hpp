#pragma once
#include <Eigen/Geometry>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <iostream>

namespace LLVM
{
using namespace Eigen;
using namespace std;
using g2o::Matrix3;
using g2o::Vector3;
using g2o::VertexSE3;
using g2o::VertexSim3Expmap;

class EdgeGICP
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  // point positions
  Vector3 pos0, pos1;

  // unit normals
  Vector3 normal0, normal1;

  // rotation matrix for normal
  Matrix3 R0, R1;

  // initialize an object
  EdgeGICP()
  {
    pos0.setZero();
    pos1.setZero();
    normal0 << 0, 0, 1;
    normal1 << 0, 0, 1;
    //makeRot();
    R0.setIdentity();
    R1.setIdentity();
  }

  // set up rotation matrix for pos0
  void makeRot0()
  {
    Vector3 y;
    y << 0, 1, 0;
    R0.row(2) = normal0;
    y = y - normal0(1) * normal0;
    y.normalize();  // need to check if y is close to 0
    R0.row(1) = y;
    R0.row(0) = normal0.cross(R0.row(1));
    //      cout << normal.transpose() << endl;
    //      cout << R0 << endl << endl;
    //      cout << R0*R0.transpose() << endl << endl;
  }

  // set up rotation matrix for pos1
  void makeRot1()
  {
    Vector3 y;
    y << 0, 1, 0;
    R1.row(2) = normal1;
    y = y - normal1(1) * normal1;
    y.normalize();  // need to check if y is close to 0
    R1.row(1) = y;
    R1.row(0) = normal1.cross(R1.row(1));
  }

  // returns a precision matrix for point-plane
  Matrix3 prec0(number_t e)
  {
    makeRot0();
    Matrix3 prec;
    prec << e, 0, 0,
        0, e, 0,
        0, 0, 1;
    return R0.transpose() * prec * R0;
  }

  // returns a precision matrix for point-plane
  Matrix3 prec1(number_t e)
  {
    makeRot1();
    Matrix3 prec;
    prec << e, 0, 0,
        0, e, 0,
        0, 0, 1;
    return R1.transpose() * prec * R1;
  }

  // return a covariance matrix for plane-plane
  Matrix3 cov0(number_t e)
  {
    makeRot0();
    Matrix3 cov;
    cov << 1, 0, 0,
        0, 1, 0,
        0, 0, e;
    return R0.transpose() * cov * R0;
  }

  // return a covariance matrix for plane-plane
  Matrix3 cov1(number_t e)
  {
    makeRot1();
    Matrix3 cov;
    cov << 1, 0, 0,
        0, 1, 0,
        0, 0, e;
    return R1.transpose() * cov * R1;
  }
};

// first two args are the measurement type, second two the connection classes
class Edge_V_V_GICP : public g2o::BaseBinaryEdge<3, EdgeGICP, VertexSE3, VertexSim3Expmap>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Edge_V_V_GICP() : pl_pl(false) {}
  Edge_V_V_GICP(const Edge_V_V_GICP* e);

  // switch to go between point-plane and plane-plane
  bool pl_pl;
  Matrix3 cov0, cov1;

  // I/O functions
  virtual bool read(std::istream& /*is*/)
  {
    cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
    return false;
  }

  virtual bool write(std::ostream& /*os*/) const
  {
    cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
    return false;
  }

  // return the error estimate as a 3-vector
  void computeError()
  {
    // from <ViewPoint> to <Point>
    const VertexSE3* vp0 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSim3Expmap* vp1 = static_cast<const VertexSim3Expmap*>(_vertices[1]);

    // get vp1 point into vp0 frame
    // could be more efficient if we computed this transform just once
    Vector3 p1;

#if 0
      if (_cnum >= 0 && 0)      // using global cache
        {
          if (_tainted[_cnum])  // set up transform
            {
              _transforms[_cnum] = vp0->estimate().inverse() * vp1->estimate();
              _tainted[_cnum] = 0;
              cout << _transforms[_cnum] << endl;
            }
          p1 = _transforms[_cnum].map(measurement().pos1); // do the transform
        }
      else
#endif
    {
      p1 = vp1->estimate().map(measurement().pos1);
      p1 = vp0->estimate().inverse() * p1;
    }

    //      cout << endl << "Error computation; points are: " << endl;
    //      cout << p0.transpose() << endl;
    //      cout << p1.transpose() << endl;

    // get their difference
    // this is simple Euclidean distance, for now
    _error = p1 - measurement().pos0;

#if 0
      cout << "vp0" << endl << vp0->estimate() << endl;
      cout << "vp1" << endl << vp1->estimate() << endl;
      cout << "e Jac Xj" << endl <<  _jacobianOplusXj << endl << endl;
      cout << "e Jac Xi" << endl << _jacobianOplusXi << endl << endl;
#endif

    if (!pl_pl) return;

    // // re-define the information matrix
    // // topLeftCorner<3,3>() is the rotation()
    // const Matrix3 transform = (vp0->estimate().inverse() * vp1->estimate()).matrix().topLeftCorner<3, 3>();
    // information() = (cov0 + transform * cov1 * transform.transpose()).inverse();
  }

#ifdef NOT_NEED
  // try analytic jacobians
#ifdef GICP_ANALYTIC_JACOBIANS
  virtual void linearizeOplus();
#endif

  // global derivative matrices
  static Matrix3 dRidx;
  static Matrix3 dRidy;
  static Matrix3 dRidz;  // differential quat matrices
#endif
};
}  // namespace LLVM
