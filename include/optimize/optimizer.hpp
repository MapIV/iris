#pragma once
#include "core/keypoints_with_normal.hpp"
#include "core/types.hpp"
#include "map/map.hpp"
#include <Eigen/Dense>

namespace vllm
{
namespace optimize
{
struct Gain {
  // for Solver
  float scale = 0;
  float latitude = 0;
  float altitude = 0;
  float smooth = 0;
};

struct Config {
  // for Optimizer
  float threshold_translation = 0;
  float threshold_rotation = 0;
  float distance_max;
  float distance_min;
  int iteration;
  Gain gain;
};

struct Outcome {
  pcl::CorrespondencesPtr correspondences = nullptr;
  Eigen::Matrix4f T_align;
};

class Optimizer
{
public:
  void setConfig(const Config& config_) { config = config_; }


  Outcome optimize(
      const std::shared_ptr<map::Map>& map_ptr,
      const KeypointsWithNormal& keypoints,  // offset
      const Eigen::Matrix4f& offset_camera,
      crrspEstimator& estimator,
      const Eigen::Matrix4f& T_initial_align
      /*, const std::list<Eigen::Matrix4f>& vllm_histroty*/);

private:
  Config config;
  crrspRejector distance_rejector;
};
}  // namespace optimize
}  // namespace vllm