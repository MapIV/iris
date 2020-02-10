#include "bridge.hpp"
#include "openvslam/data/landmark.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include "pangolin_cloud.hpp"
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr convertLandmarks(
    const std::vector<openvslam::data::landmark*>& landmarks,
    const std::set<openvslam::data::landmark*>& local_landmarks)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (landmarks.empty()) {
    return cloud;
  }

  for (const auto lm : landmarks) {
    if (!lm || lm->will_be_erased()) {
      continue;
    }
    if (local_landmarks.count(lm)) {
      continue;
    }
    const openvslam::Vec3_t pos = lm->get_pos_in_world();
    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));

    cloud->push_back(p);
  }
  for (const auto local_lm : local_landmarks) {
    if (local_lm->will_be_erased()) {
      continue;
    }
    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    pcl::PointXYZ p(
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z()));
    cloud->push_back(p);
  }

  return cloud;
}

class PangolinViewer
{
  const std::string window_name = "Pangolin";

  std::shared_ptr<pangolin::Var<std::string>> ui_state_ptr;
  pangolin::OpenGlRenderState s_cam;
  pangolin::Handler3D handler;
  pangolin::View d_cam;

public:
  PangolinViewer()
      : s_cam(pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY))),
        handler(pangolin::Handler3D(s_cam))
  {
    // setup Pangolin viewer
    pangolin::CreateWindowAndBind(window_name, 1024, 768);
    glEnable(GL_DEPTH_TEST);

    // Ensure that blending is enabled for rendering text.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    d_cam = (pangolin::CreateDisplay()
                 .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                 .SetHandler(&handler));

    // setup GUI
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));
    ui_state_ptr = std::make_shared<pangolin::Var<std::string>>("ui.state", "VLLM");
  }

  ~PangolinViewer() = default;

  void clear()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
  }

  void addPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    PangolinCloud pc(cloud);
    pc.drawPoints();
  }

  void reflesh()
  {
    pangolin::FinishFrame();
  }

  void drawState(int state)
  {
    glColor3f(1.0f, 0.0f, 0.0f);
    std::stringstream ss;
    ss << "State: ";
    switch (state) {
    case 0: ss << "NotInitialized"; break;
    case 1: ss << "Initializing"; break;
    case 2: ss << "Tracking"; break;
    case 3: ss << "Lost"; break;
    default: break;
    }
    pangolin::GlFont::I().Text(ss.str()).DrawWindow(200, 50 - 1.0f * pangolin::GlFont::I().Height());
  }
};

int main(int argc, char* argv[])
{
  BridgeOpenVSLAM bridge;
  bridge.setup(argc, argv);

  const auto frame_publisher = bridge.get_frame_publisher();
  const auto map_publisher = bridge.get_map_publisher();

  cv::namedWindow("OpenCV", cv::WINDOW_AUTOSIZE);

  PangolinViewer pangolin_viewer;

  while (true) {
    // Execute vSLAM
    bool success = bridge.execute();
    if (!success)
      break;

    std::vector<openvslam::data::landmark*> landmarks;
    std::set<openvslam::data::landmark*> local_landmarks;
    map_publisher->get_landmarks(landmarks, local_landmarks);
    auto cloud = convertLandmarks(landmarks, local_landmarks);

    // Visualize by OpenCV
    cv::imshow("OpenCV", frame_publisher->draw_frame());

    int state = static_cast<int>(frame_publisher->get_tracking_state());

    // Visualize by Pangolin
    pangolin_viewer.clear();
    pangolin_viewer.drawState(state);
    pangolin_viewer.addPointCloud(cloud);
    pangolin_viewer.reflesh();

    int key = cv::waitKey(10);
    if (key == 'q') {
      break;
    }
  }

  return 0;
}