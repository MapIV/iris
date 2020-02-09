#include "bridge.hpp"
#include "openvslam/data/landmark.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include "pangocloud.hpp"
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
    pcl::PointXYZ p(pos.x(), pos.y(), pos.z());
    cloud->push_back(p);
  }
  for (const auto local_lm : local_landmarks) {
    if (local_lm->will_be_erased()) {
      continue;
    }
    const openvslam::Vec3_t pos = local_lm->get_pos_in_world();
    pcl::PointXYZ p(pos.x(), pos.y(), pos.z());
    cloud->push_back(p);
  }

  return cloud;
}

class PangolinViewer
{
  const std::string window_name = "Pangolin";

  std::shared_ptr<pangolin::Var<double>> ui_ptr;
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
    pangolin::GetBoundWindow()->RemoveCurrent();
    pangolin::BindToContext(window_name);
    glEnable(GL_DEPTH_TEST);

    d_cam = (pangolin::CreateDisplay()
                 .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                 .SetHandler(&handler));

    // setup GUI
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));
    ui_ptr = std::make_shared<pangolin::Var<double>>("ui.double", 3, 0, 5);
  }

  void update(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);

    pangolin::glDrawColouredCube();

    PangoCloud pc(cloud);
    pc.drawPoints();

    pangolin::FinishFrame();
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

    // Visualize by Pangolin
    pangolin_viewer.update(cloud);

    int key = cv::waitKey(10);
    if (key == 'q') {
      break;
    }
  }

  pangolin::GetBoundWindow()->RemoveCurrent();
  return 0;
}
