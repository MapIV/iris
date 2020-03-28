#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>

namespace vllm
{
class TopicAnalyzer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TopicAnalyzer(const std::string& topic_csv, const std::string& imu_csv)
  {
    std::ifstream topic_ifs(topic_csv);
    std::ifstream imu_ifs(imu_csv);

    std::string line;
    while (std::getline(topic_ifs, line)) {
      Eigen::Vector3f acc = Eigen::Vector3f::Zero();
      Eigen::Vector3f omega = Eigen::Vector3f::Zero();

      if (line == "image") {
        is_topic_video.push_back(true);
      } else {
        is_topic_video.push_back(false);

        std::string imu;
        std::getline(imu_ifs, imu);
        std::istringstream ss(imu);

        float x, y, z, roll, pitch, yaw;
        ss >> x >> y >> z >> roll >> pitch >> yaw;
        acc << x, y, z;
        omega << roll, pitch, yaw;
      }

      acc_data.push_back(acc);
      omega_data.push_back(omega);
    }
  }

  bool isTopicVideo(unsigned int t) const { return is_topic_video.at(t); }
  Eigen::Vector3f getAcc(unsigned int t) const { return acc_data.at(t); }
  Eigen::Vector3f getOmega(unsigned int t) const { return omega_data.at(t); }

private:
  std::vector<Eigen::Vector3f> omega_data;
  std::vector<Eigen::Vector3f> acc_data;
  std::vector<bool> is_topic_video;
};
}  // namespace vllm