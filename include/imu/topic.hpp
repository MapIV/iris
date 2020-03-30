#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>

namespace vllm
{
struct ImuMessage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  unsigned long ns;
  Eigen::Vector3f acc;
  Eigen::Vector3f omega;
};
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
      ImuMessage msg;

      if (line == "image") {
        is_topic_video.push_back(true);
      } else {
        is_topic_video.push_back(false);

        std::string imu;
        std::getline(imu_ifs, imu);
        std::istringstream ss(imu);

        float x, y, z, roll, pitch, yaw;
        ss >> msg.ns >> x >> y >> z >> roll >> pitch >> yaw;
        msg.acc << x, y, z;
        msg.omega << roll, pitch, yaw;
      }

      data.push_back(msg);
    }
  }

  bool isTopicVideo(unsigned int t) const { return is_topic_video.at(t); }
  ImuMessage getImuMessage(unsigned int t) const { return data.at(t); }

private:
  std::vector<ImuMessage> data;
  std::vector<bool> is_topic_video;
};
}  // namespace vllm