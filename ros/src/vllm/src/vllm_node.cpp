#include "core/config.hpp"
#include "map/map.hpp"
#include "system/system.hpp"
#include "viewer/pangolin_viewer.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <popl.hpp>

void mono_tracking(
    // const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
    //   const std::string& mask_img_path, const bool eval_log, const std::string& map_db_path
)
{
  //   // load the mask image
  //   const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

  //   // build a SLAM system
  //   openvslam::system SLAM(cfg, vocab_file_path);
  //   // startup the SLAM process
  //   SLAM.startup();

  //   // create a viewer object
  //   // and pass the frame_publisher and the map_publisher
  // #ifdef USE_PANGOLIN_VIEWER
  //   pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
  // #elif USE_SOCKET_PUBLISHER
  //   socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
  // #endif

  //   std::vector<double> track_times;
  //   const auto tp_0 = std::chrono::steady_clock::now();

  //   // initialize this node
  //   const ros::NodeHandle nh;
  //   image_transport::ImageTransport it(nh);

  //   // run the SLAM as subscriber
  //   image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, [&](const sensor_msgs::ImageConstPtr& msg) {
  //     const auto tp_1 = std::chrono::steady_clock::now();
  //     const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0).count();

  //     // input the current frame and estimate the camera pose
  //     SLAM.feed_monocular_frame(cv_bridge::toCvShare(msg, "bgr8")->image, timestamp, mask);

  //     const auto tp_2 = std::chrono::steady_clock::now();

  //     const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
  //     track_times.push_back(track_time);
  //   });

  // run the viewer in another thread
  // #ifdef USE_PANGOLIN_VIEWER
  //   std::thread thread([&]() {
  //     viewer.run();
  //     if (SLAM.terminate_is_requested()) {
  //       // wait until the loop BA is finished
  //       while (SLAM.loop_BA_is_running()) {
  //         std::this_thread::sleep_for(std::chrono::microseconds(5000));
  //       }
  //       ros::shutdown();
  //     }
  //   });

  ros::spin();

  // automatically close the viewer
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vllm_node");

  // // check validness of options
  // if (help->is_set()) {
  //   std::cerr << op << std::endl;
  //   return EXIT_FAILURE;
  // }
  // if (!vocab_file_path->is_set() || !setting_file_path->is_set()) {
  //   std::cerr << "invalid arguments" << std::endl;
  //   std::cerr << std::endl;
  //   std::cerr << op << std::endl;
  //   return EXIT_FAILURE;
  // }

  // // setup logger
  // spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  // if (debug_mode->is_set()) {
  //   spdlog::set_level(spdlog::level::debug);
  // } else {
  //   spdlog::set_level(spdlog::level::info);
  // }

  // // load configuration
  // std::shared_ptr<openvslam::config> cfg;
  // try {
  //   cfg = std::make_shared<openvslam::config>(setting_file_path->value());
  // } catch (const std::exception& e) {
  //   std::cerr << e.what() << std::endl;
  //   return EXIT_FAILURE;
  // }


  // // run tracking
  // if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
  //   mono_tracking(cfg, vocab_file_path->value(), mask_img_path->value(), eval_log->is_set(), map_db_path->value());
  // } else {
  //   throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
  // }

  return EXIT_SUCCESS;
}
