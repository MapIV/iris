#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

class Aligner
{
public:
  Aligner()
  {
    optimizer.setVerbose(true);

    // variable-size block solver
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
    optimizer.setAlgorithm(solver);
  }

  void estimate(
      const pcl::PointCloud<pcl::PointXYZ>& source,
      const pcl::PointCloud<pcl::PointXYZ>& target,
      Eigen::Matrix4d& transform_matrix) const
  {
  }

private:
  g2o::SparseOptimizer optimizer;
};