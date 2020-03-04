#include "aligner.hpp"
#include "types_gicp.hpp"
#include "util.hpp"

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
Eigen::Matrix4f Aligner::estimate6DoF(
    Eigen::Matrix4f& T,
    const pcl::PointCloud<pcl::PointXYZ>& source,
    const pcl::PointCloud<pcl::PointXYZ>& target,
    const pcl::Correspondences& correspondances,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
    const pcl::PointCloud<pcl::Normal>::Ptr& source_normals)
{
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  setVertexSE3(optimizer, T);
  setEdge6DoFGICP(optimizer, source, target, correspondances, target_normals, source_normals);

  // execute
  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  optimizer.optimize(10);

  // construct output matrix
  g2o::VertexSE3* optimized = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
  Eigen::Matrix3f R = optimized->estimate().rotation().matrix().cast<float>();
  Eigen::Vector3f t = optimized->estimate().translation().cast<float>();

  Eigen::Matrix4f _T = Eigen::Matrix4f::Identity();
  _T.topLeftCorner(3, 3) = R;
  _T.topRightCorner(3, 1) = t;
  return _T;
}

void Aligner::setVertexSE3(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& T)
{
  // set up rotation and translation for this node
  Eigen::Vector3d t(0, 0, 0);
  Eigen::Quaterniond q;
  q.setIdentity();
  Eigen::Isometry3d camera;
  Eigen::Matrix3d R = T.topLeftCorner(3, 3).cast<double>();
  camera = Eigen::Quaterniond(R);
  camera.translation() = T.topRightCorner(3, 1).cast<double>();

  // set up initial parameter
  g2o::VertexSE3* vc = new g2o::VertexSE3();
  vc->setEstimate(camera);
  vc->setId(0);

  // add to optimizer
  optimizer.addVertex(vc);
}

void Aligner::setEdge6DoFGICP(
    g2o::SparseOptimizer& optimizer,
    const pcl::PointCloud<pcl::PointXYZ>& source,
    const pcl::PointCloud<pcl::PointXYZ>& target,
    const pcl::Correspondences& correspondances,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
    const pcl::PointCloud<pcl::Normal>::Ptr& source_normals)
{
  // get Vertex
  g2o::VertexSE3* vp0 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
  const Eigen::Matrix3d R = vp0->estimate().rotation().matrix();

  for (const pcl::Correspondence& cor : correspondances) {
    // new edge with correct cohort for caching
    vllm::Edge_SE3_GICP* e;
    e = new vllm::Edge_SE3_GICP(source_normals != nullptr);
    e->setVertex(0, vp0);  // set viewpoint

    // calculate the relative 3D position of the point
    Eigen::Vector3f pt0, pt1;
    pt0 = target.at(cor.index_match).getArray3fMap();
    pt1 = source.at(cor.index_query).getArray3fMap();

    vllm::EdgeGICP meas;
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
      e->cov0 = meas.cov0(0.01f);
      e->cov1 = meas.cov1(0.01f);
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

  // add a Regularization Edge of Pitch
  Edge_RollPitch_Regularizer* e = new Edge_RollPitch_Regularizer(pitch_gain);
  e->setVertex(0, vp0);
  e->information().setIdentity();
  e->setMeasurement(0.0);
  optimizer.addEdge(e);
}


}  // namespace vllm