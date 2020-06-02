#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace iris
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
    }

    // clang-format off
    iteration =     node["Iris.iteration"].as<int>();
    scale_gain =    node["Iris.scale_gain"].as<float>();
    latitude_gain = node["Iris.latitude_gain"].as<float>();
    altitude_gain = node["Iris.altitude_gain"].as<float>();
    smooth_gain =   node["Iris.smooth_gain"].as<float>();

    distance_min =         node["Iris.distance_min"].as<float>();
    distance_max =         node["Iris.distance_max"].as<float>();
    converge_translation = node["Iris.converge_translation"].as<float>();
    converge_rotation =    node["Iris.converge_rotation"].as<float>();

    normal_search_leaf = node["Map.normal_search_leaf"].as<float>();
    voxel_grid_leaf =    node["Map.voxel_grid_leaf"].as<float>();
    submap_grid_leaf =   node["Map.submap_grid_leaf"].as<float>();
    // clang-format on
  }

  float distance_min, distance_max;
  float scale_gain, latitude_gain, smooth_gain, altitude_gain;
  int iteration;

  float converge_translation, converge_rotation;
  float normal_search_leaf, voxel_grid_leaf, submap_grid_leaf;

  Eigen::Matrix4f T_init;
};
}  // namespace iris