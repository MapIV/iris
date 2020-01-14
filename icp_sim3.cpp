// #include <Eigen/StdVector>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/icp/types_icp.h>
#include <iostream>
#include <random>
#include <stdint.h>

int main()
{
  // noise in position[m]
  double euc_noise = 0.01;
  //  double outlier_ratio = 0.1;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  // variable-size block solver
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  // point cloud in R^3
  constexpr int N = 1000;
  std::vector<Eigen::Vector3d> true_points;
  for (size_t i = 0; i < N; ++i) {
    true_points.push_back(Eigen::Vector3d(
        (g2o::Sampler::uniformRand(0.0, 1.0) - 0.5) * 3,
        g2o::Sampler::uniformRand(0.0, 1.0) - 0.5,
        g2o::Sampler::uniformRand(0.0, 1.0) + 10));
  }

  // set up two poses
  int vertex_id = 0;
  for (size_t i = 0; i < 2; ++i) {
    // set up rotation and translation for this node
    Eigen::Vector3d t(0, 0, static_cast<int>(i));
    Eigen::Quaterniond q;
    q.setIdentity();

    Eigen::Isometry3d cam;  // camera pose
    cam = q;
    cam.translation() = t;

    // set up node
    g2o::VertexSE3* vc = new g2o::VertexSE3();
    vc->setEstimate(cam);
    vc->setId(vertex_id);  // vertex id

    std::cerr << t.transpose() << " | " << q.coeffs().transpose() << std::endl;

    // set first cam pose fixed
    if (i == 0)
      vc->setFixed(true);

    // add to optimizer
    optimizer.addVertex(vc);
    vertex_id++;
  }

  // set up point matches
  for (size_t i = 0; i < true_points.size(); ++i) {
    // get two poses
    g2o::VertexSE3* vp0 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
    g2o::VertexSE3* vp1 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);

    // calculate the relative 3D position of the point
    Eigen::Vector3d pt0, pt1;
    pt0 = vp0->estimate().inverse() * true_points[i];
    pt1 = vp1->estimate().inverse() * true_points[i];

    // add in noise
    pt0 += Eigen::Vector3d(
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise));
    pt1 += Eigen::Vector3d(
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise));

    // form edge, with normals in varioius positions
    Eigen::Vector3d nm0, nm1;
    nm0 << 0, static_cast<double>(i), 1;
    nm1 << 0, static_cast<double>(i), 1;
    nm0.normalize();
    nm1.normalize();

    // new edge with correct cohort for caching
    g2o::Edge_V_V_GICP* e = new g2o::Edge_V_V_GICP();

    e->setVertex(0, vp0);  // first viewpoint
    e->setVertex(1, vp1);  // second viewpoint

    g2o::EdgeGICP meas;
    meas.pos0 = pt0;
    meas.pos1 = pt1;
    meas.normal0 = nm0;
    meas.normal1 = nm1;

    e->setMeasurement(meas);

    meas = e->measurement();

    // e->information() = meas.prec0(0.01);// use this for point-plane
    e->information().setIdentity();  // use this for point-point


    // Kernel Huber(default delta = 1.0)
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);

    optimizer.addEdge(e);
  }

  // move second cam off of its true position
  g2o::VertexSE3* vc = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);
  Eigen::Isometry3d cam = vc->estimate();
  cam.translation() = Eigen::Vector3d(0, 0, 0.2);
  vc->setEstimate(cam);

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  std::cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << std::endl;

  optimizer.setVerbose(true);
  optimizer.optimize(5);

  // clang-format off
  std::cout << std::endl
            << "Second vertex should be near 0,0,1" 
            << std::endl;
  std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second)->estimate().translation().transpose()
            << std::endl;
  std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second)->estimate().translation().transpose()
            << std::endl;
  // clang-format on
}