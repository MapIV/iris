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

#include "optimize/aligner.hpp"
#include "core/util.hpp"
#include "optimize/types_gicp.hpp"
#include "optimize/types_restriction.hpp"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace iris
{
namespace optimize
{
Eigen::Matrix4f Aligner::estimate7DoF(
    Eigen::Matrix4f& T,
    const pcXYZIN::Ptr& source_clouds,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondances,
    const Eigen::Matrix4f& offset_camera,
    const std::list<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&,
    const double ref_scale,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals)
{
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  setVertexSim3(optimizer, T);
  setEdge7DoFGICP(optimizer, source_clouds, target, correspondances, offset_camera.topRightCorner(3, 1), target_normals);
  setEdgeRestriction(optimizer, offset_camera, T, ref_scale);

  // execute
  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  optimizer.optimize(5);

  // construct output matrix
  g2o::VertexSim3Expmap* optimized = dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(0)->second);
  float scale = static_cast<float>(optimized->estimate().scale());
  Eigen::Matrix3f R = optimized->estimate().rotation().matrix().cast<float>();
  Eigen::Vector3f t = optimized->estimate().translation().cast<float>();
  std::cout << "scale= \033[31m" << scale << "\033[m" << std::endl;

  T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = scale * R;
  T.topRightCorner(3, 1) = t;

  return T;
}

void Aligner::setVertexSim3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T)
{
  // set up rotation and translation for this node
  Eigen::Vector3d t = T.topRightCorner(3, 1).cast<double>();
  Eigen::Matrix3d sR = T.topLeftCorner(3, 3).cast<double>();
  double scale = util::getScale(sR.cast<float>());
  Eigen::Quaterniond q = Eigen::Quaterniond(sR / scale);
  g2o::Sim3 sim3(q, t, scale);

  // set up initial parameter
  g2o::VertexSim3Expmap* vc = new g2o::VertexSim3Expmap();
  vc->setEstimate(sim3);
  vc->setId(0);

  // add to optimizer
  optimizer.addVertex(vc);
}

void Aligner::setEdge7DoFGICP(
    g2o::SparseOptimizer& optimizer,
    const pcXYZIN::Ptr& source_clouds,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondances,
    const Eigen::Vector3f&,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals)
{
  // get Vertex
  g2o::VertexSim3Expmap* vp0 = dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(0)->second);
  const Eigen::Matrix3d R = vp0->estimate().rotation().matrix();

  for (const pcl::Correspondence& cor : *correspondances) {
    // new edge with correct cohort for caching
    Edge_Sim3_GICP* e = new Edge_Sim3_GICP(true);
    e->setVertex(0, vp0);  // set viewpoint

    // calculate the relative 3D position of the point
    Eigen::Vector3f pt0, pt1;
    pt0 = target->at(cor.index_match).getVector3fMap();
    pt1 = source_clouds->at(cor.index_query).getVector3fMap();
    float weight = source_clouds->at(cor.index_query).intensity;

    EdgeGICP meas;
    meas.weight = weight;
    // meas.weight = 1.0f / ((camera - pt1).norm() + 1.0f);
    meas.pos0 = pt0.cast<double>();
    meas.pos1 = pt1.cast<double>();

    e->setMeasurement(meas);

    Eigen::Vector3f n0 = target_normals->at(cor.index_match).getNormalVector3fMap();
    Eigen::Vector3f n1 = source_clouds->at(cor.index_query).getNormalVector3fMap();

    // sometime normal0 has NaN
    if (std::isfinite(n0.x())) {
      meas.normal0 = n0.cast<double>();
      e->cov0 = meas.cov0(0.05f);  // target
    } else {
      e->cov0 = meas.cov0(1.0f);  // target
    }
    meas.normal1 = n1.cast<double>();
    e->cov1 = meas.cov1(1.00f);  // source
    // e->information() = (e->cov0 + R * e->cov1 * R.transpose()).inverse();
    e->information() = (e->cov0).inverse();


    // set Huber kernel (default delta = 1.0)
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    optimizer.addEdge(e);
  }
}

void Aligner::setEdgeRestriction(
    g2o::SparseOptimizer& optimizer,
    const Eigen::Matrix4f& offset_camera,
    const Eigen::Matrix4f&,
    double ref_scale)
{
  g2o::VertexSim3Expmap* vp0 = dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(0)->second);

  // Add a scale edge
  {
    Edge_Scale_Restriction* e = new Edge_Scale_Restriction(scale_gain);
    e->setVertex(0, vp0);
    e->information().setIdentity();
    e->setMeasurement(ref_scale);
    optimizer.addEdge(e);
  }

  // Add an altitude edge
  // {
  //   Edge_Altitude_Restriction* e = new Edge_Altitude_Restriction(altitude_gain);
  //   e->setVertex(0, vp0);
  //   e->information().setIdentity();
  //   e->setMeasurement(offset_camera.topRightCorner(3, 1).cast<double>());
  //   optimizer.addEdge(e);
  // }

  // Add a latitude edge
  // {
  //   Eigen::Matrix3f R = util::normalizeRotation(offset_camera);
  //   Edge_Latitude_Restriction* e = new Edge_Latitude_Restriction(R.cast<double>(), latitude_gain);
  //   e->setVertex(0, vp0);
  //   e->information().setIdentity();
  //   e->setMeasurement(0.0);
  //   optimizer.addEdge(e);
  // }

  //  TODO:
  // // add a const velocity Model Constraint Edge of Scale
  // {
  //   Edge_Smooth_Restriction* e = new Edge_Smooth_Restriction(smooth_gain);
  //   e->setVertex(0, vp0);
  //   e->information().setIdentity();
  //   VelocityModel model;
  //   model.camera_pos = offset_camera.topRightCorner(3, 1).cast<double>();

  //   model.old_pos = itr1->topRightCorner(3, 1).cast<double>();
  //   model.older_pos = itr2->topRightCorner(3, 1).cast<double>();
  //   e->setMeasurement(model);
  //   optimizer.addEdge(e);
  // }
}

}  // namespace optimize
}  // namespace iris
