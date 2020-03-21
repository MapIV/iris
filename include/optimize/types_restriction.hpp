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
};

class Edge_Scale_Restriction : public g2o::BaseUnaryEdge<1, double, VertexSim3Expmap>
{
private:
  double gain = 1.0;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge_Scale_Restriction(double gain = 1.0) : gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

class Edge_Smooth_Restriction : public g2o::BaseUnaryEdge<3, VelocityModel, VertexSim3Expmap>
{
private:
  double gain = 1.0;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge_Smooth_Restriction(double gain = 1.0) : gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

class Edge_Latitude_Restriction : public g2o::BaseUnaryEdge<1, double, VertexSim3Expmap>
{
private:
  double gain = 1.0;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge_Latitude_Restriction(double gain = 1.0) : gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

class Edge_Altitude_Restriction : public g2o::BaseUnaryEdge<1, Eigen::Vector3d, VertexSim3Expmap>
{
private:
  double gain = 1.0;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge_Altitude_Restriction(double gain = 1.0) : gain(gain) {}

  virtual bool read(std::istream&) { return false; }
  virtual bool write(std::ostream&) const { return false; }
  void computeError();
};

}  // namespace optimize
}  // namespace vllm
