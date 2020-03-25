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

namespace vllm
{
namespace optimize
{
Eigen::Matrix4f Aligner::estimate7DoF(
    Eigen::Matrix4f& T,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondances,
    const Eigen::Matrix4f& offset_camera,
    const std::list<Eigen::Matrix4f>& history,
    const std::vector<float>& weights,
    const pcl::PointCloud<pcl::Normal>::Ptr& source_normals,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals)
{
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  setVertexSim3(optimizer, T);
  setEdge7DoFGICP(optimizer, source, target, correspondances, weights, target_normals, source_normals);
  setEdgeRestriction(optimizer, offset_camera, history);

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
  Eigen::Matrix3d R = T.topLeftCorner(3, 3).cast<double>();
  Eigen::Quaterniond q = Eigen::Quaterniond(R);
  double scale = vllm::getScale(R.cast<float>());
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
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondances,
    const std::vector<float>& weights,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
    const pcl::PointCloud<pcl::Normal>::Ptr& source_normals)
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
    pt0 = target->at(cor.index_match).getArray3fMap();
    pt1 = source->at(cor.index_query).getArray3fMap();
    float weight = weights.at(cor.index_query);

    EdgeGICP meas;
    meas.weight = weight;
    meas.pos0 = pt0.cast<double>();
    meas.pos1 = pt1.cast<double>();

    e->setMeasurement(meas);
    e->information().setIdentity();
    if (source_normals) {
      Eigen::Vector3f n0 = target_normals->at(cor.index_match).getNormalVector3fMap();
      Eigen::Vector3f n1 = source_normals->at(cor.index_query).getNormalVector3fMap();

      // sometime normal0 has NaN
      if (std::isfinite(n0.x())) meas.normal0 = n0.cast<double>();
      meas.normal1 = n1.cast<double>();
      e->cov0 = meas.cov0(0.01f);  // target
      e->cov1 = meas.cov1(0.05f);  // source
      e->information() = (e->cov0 + R * e->cov1 * R.transpose()).inverse();

    } else if (target_normals) {
      Eigen::Vector3f n = target_normals->at(cor.index_match).getNormalVector3fMap();
      if (std::isfinite(n.x())) {  // sometime n has NaN
        meas.normal0 = n.cast<double>();
        e->information() = meas.prec0(0.01f);
      }
    }

    // set Huber kernel (default delta = 1.0)
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    optimizer.addEdge(e);
  }
}

void Aligner::setEdgeRestriction(
    g2o::SparseOptimizer& optimizer,
    const Eigen::Matrix4f& offset_camera,
    const std::list<Eigen::Matrix4f>& history)
{
  g2o::VertexSim3Expmap* vp0 = dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(0)->second);

  // const unsigned int DT = 2;
  // auto itr1 = std::next(history.begin(), DT);
  // auto itr2 = std::next(itr1, DT);

  // Add a scale edge
  {
    Edge_Scale_Restriction* e = new Edge_Scale_Restriction(scale_gain);
    e->setVertex(0, vp0);
    e->information().setIdentity();
    std::vector<double> scales;
    for (const Eigen::Matrix4f& T : history)
      scales.push_back(static_cast<double>(getScale(T)));
    e->setMeasurement(scales);
    optimizer.addEdge(e);
  }

  // Add an altitude edge
  {
    Edge_Altitude_Restriction* e = new Edge_Altitude_Restriction(altitude_gain);
    e->setVertex(0, vp0);
    e->information().setIdentity();
    e->setMeasurement(offset_camera.topRightCorner(3, 1).cast<double>());
    optimizer.addEdge(e);
  }

  // Add a latitude edge
  {
    Edge_Latitude_Restriction* e = new Edge_Latitude_Restriction(latitude_gain);
    e->setVertex(0, vp0);
    e->information().setIdentity();
    e->setMeasurement(0.0);
    optimizer.addEdge(e);
  }

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
}  // namespace vllm
