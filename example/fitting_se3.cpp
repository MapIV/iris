#include <Eigen/Dense>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

Eigen::Matrix3d hat(const Eigen::Vector3d& w)
{
  Eigen::Matrix3d S;
  S << 0, -w(2), w(1),
      w(2), 0, -w(0),
      -w(1), w(0), 0;
  return S;
}

double error(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B)
{
  double theta = (A - B).topLeftCorner(3, 3).norm();
  double p = (A - B).topRightCorner(3, 1).norm();
  return theta + p;
}

namespace so3
{
Eigen::Vector3d log(const Eigen::Matrix3d& R)
{
  Eigen::Vector3d xi = Eigen::Vector3d::Zero();
  double w_length = static_cast<double>(std::acos((R.trace() - 1.0f) * 0.5f));
  if (w_length > 1e-6f) {
    Eigen::Vector3d tmp;
    tmp << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    xi = 1.0f / (2.0f * static_cast<double>(std::sin(w_length))) * tmp * w_length;
  }
  return xi;
}

Eigen::Matrix3d exp(const Eigen::Vector3d& w)
{
  double theta = w.norm();
  Eigen::Vector3d axis = w.normalized();

  double cos = std::cos(theta);
  double sin = std::sin(theta);

  Eigen::Matrix3d tmp1 = cos * Eigen::Matrix3d::Identity();
  Eigen::Matrix3d tmp2 = (1 - cos) * (axis * axis.transpose());
  Eigen::Matrix3d tmp3 = sin * hat(axis);
  return tmp1 + tmp2 + tmp3;
}
}  // namespace so3

namespace se3
{
Eigen::VectorXd log(const Eigen::Matrix4d& T)
{
  Eigen::Matrix3d R = T.topLeftCorner(3, 3);
  Eigen::Vector3d t = T.topRightCorner(3, 1);
  Eigen::Vector3d w = so3::log(R);

  Eigen::Matrix3d hat_w = hat(w);

  double w_length = w.norm();
  Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity();
  if (w_length > 1e-6) {
    V_inv = Eigen::Matrix3d::Identity() - 0.5 * hat_w
            + (1.0 - (w_length * std::cos(w_length * 0.5)) / (2.0 * std::sin(w_length * 0.5)))
                  * (hat_w * hat_w) / (w_length * w_length);
  }
  Eigen::Vector3d v = V_inv * t;

  Eigen::VectorXd xi(6);
  xi.topRows(3) = v;
  xi.bottomRows(3) = w;
  return xi;
}
Eigen::Matrix4d exp(const Eigen::VectorXd& xi)
{
  Eigen::Vector3d v = xi.topRows(3);
  Eigen::Vector3d w = xi.bottomRows(3);
  Eigen::Matrix3d hat_w = hat(w);
  double w_length = w.norm();

  // rotation
  Eigen::Matrix3d R = so3::exp(w);

  // translation
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  if (w_length > 1e-6) {
    Eigen::Matrix3d V(Eigen::Matrix3d::Identity()
                      + hat_w * (1.0 - std::cos(w_length)) / (w_length * w_length)
                      + (hat_w * hat_w) * (w_length - static_cast<double>(std::sin(w_length))) / (w_length * w_length * w_length));
    t = V * v;
  } else {
    t = v;
  }

  // pose
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner(3, 3) = R;
  T.topRightCorner(3, 1) = t;
  return T;
}

// {6x1,6x1} => 6x1
Eigen::VectorXd concatenate(const Eigen::VectorXd& xi0, const Eigen::VectorXd& xi1)
{
  return log(exp(xi0) * exp(xi1));
}

}  // namespace se3


cv::Point3f conv(const Eigen::Vector3d& v)
{
  return cv::Point3f(v(0), v(1), v(2));
}


namespace solver
{
class Model
{
public:
  Eigen::Matrix4d T;
  float time;
};

class Vertex : public g2o::BaseVertex<6, Eigen::VectorXd>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Vertex()
  {
  }

  virtual bool read(std::istream& /*is*/)
  {
    return false;
  }

  virtual bool write(std::ostream& /*os*/) const
  {
    return false;
  }

  virtual void setToOriginImpl()
  {
  }

  virtual void oplusImpl(const double* update)
  {
    Eigen::VectorXd::ConstMapType v(update, 6);
    _estimate += v;
  }
};

class Edge : public g2o::BaseUnaryEdge<12, Model, Vertex>
{
public:
  Edge() {}
  Edge(const Edge* e);

  virtual bool read(std::istream&)
  {
    return false;
  }
  virtual bool write(std::ostream&) const
  {
    return false;
  }
  void computeError()
  {
    Eigen::Matrix4d obs = measurement().T;
    const float time = measurement().time;

    const Vertex* vp0 = static_cast<const Vertex*>(_vertices[0]);
    Eigen::VectorXd xi = vp0->estimate();

    Eigen::Matrix4d est = se3::exp(xi);
    Eigen::Matrix4d dT = (obs - est);

    _error.topRows(3) = dT.topRightCorner(3, 1).cast<double>();
    for (int i = 0; i < 9; i++)
      _error(3 + i) = dT.data()[i];
    std::cout << time << " " << _error.transpose() << std::endl;
  }
};

void setVertex(g2o::SparseOptimizer& optimizer)
{
  // set up rotation and translation for this node
  Eigen::VectorXd xi = Eigen::VectorXd::Ones(6);

  // set up initial parameter
  solver::Vertex* vc = new solver::Vertex();
  vc->setEstimate(xi);
  vc->setId(0);

  // add to optimizer
  optimizer.addVertex(vc);
}

void setEdge(g2o::SparseOptimizer& optimizer, const std::vector<Eigen::Matrix4d>& poses)
{
  // get Vertex
  solver::Vertex* vp0 = dynamic_cast<solver::Vertex*>(optimizer.vertices().find(0)->second);

  for (size_t i = 1; i < poses.size(); i++) {
    const Eigen::Matrix4d pose = poses.at(i);
    const Eigen::Matrix4d pre = poses.at(i - 1);
    solver::Edge* e = new solver::Edge;
    e->setVertex(0, vp0);  // set viewpoint

    Model model;
    model.T = pose * pre.inverse();
    model.time = static_cast<float>(i);

    e->setMeasurement(model);
    e->information().setIdentity();

    // set Huber kernel (default delta = 1.0)
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    optimizer.addEdge(e);
  }
}

}  // namespace solver

int main()
{
  srand((unsigned int)time(0));

  // Make random pose
  Eigen::VectorXd seed = Eigen::VectorXd::Random(6);
  Eigen::Matrix4d T = se3::exp(seed);
  Eigen::VectorXd xi = se3::log(T);

  // Make observation
  const int N = 4;
  std::vector<Eigen::Matrix4d> poses;
  poses.push_back(Eigen::Matrix4d::Identity());
  for (int i = 1; i < N; i++) {
    Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
    double r = (rand() % 10 - 5) / 10.0;
    tmp = se3::exp((i + r) * xi);
    poses.push_back(tmp);
  }

  // Construct optimizer
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);
  solver::setVertex(optimizer);
  solver::setEdge(optimizer, poses);

  // Optimize
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  optimizer.optimize(15);

  // get optimal
  solver::Vertex* optimized = dynamic_cast<solver::Vertex*>(optimizer.vertices().find(0)->second);
  Eigen::VectorXd optimal = optimized->estimate();
  std::cout << "ans: " << xi.transpose() << std::endl;
  std::cout << "opt: " << optimal.transpose() << std::endl;

  // Viewer
  static bool loop = true;
  cv::viz::Viz3d viz_window("view");
  viz_window.showWidget("coordinate", cv::viz::WCoordinateSystem(0.5));
  viz_window.setWindowSize(cv::Size(640, 480));
  viz_window.registerKeyboardCallback([](const cv::viz::KeyboardEvent&, void*) -> void { loop = false; }, &viz_window);

  // Draw observation
  Eigen::Vector4d origin;
  origin << 0, 0, 0.5, 1;
  for (int i = 0; i < N; i++) {
    Eigen::Matrix4d pose = poses.at(i);
    Eigen::Vector4d after = pose * origin;
    cv::viz::WArrow arrow(conv(pose.topRightCorner(3, 1)), conv(after.topRows(3)), 0.01, cv::viz::Color::red());
    viz_window.showWidget("obs" + std::to_string(i), arrow);
  }

  // Draw intermidiation
  for (int i = 0; i < 10 * N; i++) {
    Eigen::Matrix4d tmp = se3::exp(optimal * static_cast<double>(i) / 10.0);
    Eigen::Vector4d after = tmp * origin;
    cv::viz::WArrow arrow(conv(tmp.topRightCorner(3, 1)), conv(after.topRows(3)), 0.01, cv::viz::Color::yellow());
    viz_window.showWidget("est" + std::to_string(i), arrow);
  }

  while (loop) {
    viz_window.spinOnce(1, true);
  }
}