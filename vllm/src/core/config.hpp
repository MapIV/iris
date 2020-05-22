#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace vllm
{
struct Config {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Config() {}

  Config(const std::string& yaml_file)
  {
    init(yaml_file);
  }

  void init(const std::string& yaml_file)
  {
    YAML::Node node;
    try {
      node = YAML::LoadFile(yaml_file);
    } catch (YAML::ParserException& e) {
      std::cout << e.what() << "\n";
      std::cout << "can not open " << yaml_file << std::endl;
      exit(1);
    }

    {
      Eigen::Vector3f t(node["Init.transform"].as<std::vector<float>>().data());
      Eigen::Vector3f n(node["Init.normal"].as<std::vector<float>>().data());
      Eigen::Vector3f u(node["Init.upper"].as<std::vector<float>>().data());
      float s = node["Init.scale"].as<float>();

      Eigen::Matrix3f R;
      n.normalize();
      u.normalize();
      R.row(2) = n;
      R.row(1) = (n.dot(u) * n - u).normalized();  // Gram–Schmidt orthonormalization
      R.row(0) = R.row(1).cross(R.row(2));

      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.topLeftCorner(3, 3) = s * R.transpose();
      T.topRightCorner(3, 1) = t;
      T_init = T;
      std::cout << T << std::endl;
    }

    // clang-format off
    iteration =     node["VLLM.iteration"].as<int>();
    scale_gain =    node["VLLM.scale_gain"].as<float>();
    latitude_gain = node["VLLM.latitude_gain"].as<float>();
    altitude_gain = node["VLLM.altitude_gain"].as<float>();
    smooth_gain =   node["VLLM.smooth_gain"].as<float>();
    // clang-format on

    distance_min = node["VLLM.distance_min"].as<float>();
    distance_max = node["VLLM.distance_max"].as<float>();
    converge_translation = node["VLLM.converge_translation"].as<float>();
    converge_rotation = node["VLLM.converge_rotation"].as<float>();

    normal_search_leaf = node["Map.normal_search_leaf"].as<float>();
    voxel_grid_leaf = node["Map.voxel_grid_leaf"].as<float>();
    submap_grid_leaf = node["Map.submap_grid_leaf"].as<float>();
  }

  float distance_min, distance_max;
  float scale_gain, latitude_gain, smooth_gain, altitude_gain;
  int iteration;

  float converge_translation;
  float converge_rotation;

  float normal_search_leaf;
  float voxel_grid_leaf;
  float submap_grid_leaf;

  Eigen::Matrix4f T_init;
};
}  // namespace vllm