#pragma once
#include "pangolin_cloud.hpp"
#include <pangolin/pangolin.h>

struct Color {
  float r;
  float g;
  float b;
  float size;
  Color() { r = g = b = size = 1.0f; }
  Color(float r, float g, float b, float s) : r(r), g(g), b(b), size(s) {}
};

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
            pangolin::ModelViewLookAt(0, -2, -2, 0, 0, 0, pangolin::AxisNegY))),
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

  void drawGridLine()
  {
    Eigen::Matrix4f origin;
    // clang-format off
    origin << 
       1,  0, 0, 0,
       0,  1, 0, 0, 
       0,  0, 1, 0, 
       0,  0, 0, 1;
    // clang-format on
    glPushMatrix();
    glMultTransposeMatrixf(origin.data());

    glLineWidth(1);
    glColor3f(0.3f, 0.3f, 0.3f);

    glBegin(GL_LINES);
    constexpr float interval_ratio = 0.1f;
    constexpr float grid_min = -100.0f * interval_ratio;
    constexpr float grid_max = 100.0f * interval_ratio;
    for (float x = -10.f; x <= 10.f; x += 1.0f) {
      draw_line(x * 10.0f * interval_ratio, grid_min, 0, x * 10.0f * interval_ratio, grid_max, 0);
    }
    for (float y = -10.f; y <= 10.f; y += 1.0f) {
      draw_line(grid_min, y * 10.0f * interval_ratio, 0, grid_max, y * 10.0f * interval_ratio, 0);
    }
    glEnd();
    glPopMatrix();

    // coordinate axis
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    constexpr float axis = 1.0f;
    glColor3f(1.0f, 0.0f, 0.0f);
    draw_line(0, 0, 0, axis, 0, 0);
    glColor3f(0.0f, 1.0f, 0.0f);
    draw_line(0, 0, 0, 0, axis, 0);
    glColor3f(0.0f, 0.0f, 1.0f);
    draw_line(0, 0, 0, 0, 0, axis);
    glEnd();
  }

  void addPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Color& color) const
  {
    glColor3f(color.r, color.g, color.b);
    glPointSize(color.size);

    PangolinCloud pc(cloud);
    pc.drawPoints();
  }

  void swap() const
  {
    pangolin::FinishFrame();
  }

  void drawState(int state) const
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

  void addCamera(const openvslam::Mat44_t& cam_pose_wc, const Color& color) const
  {
    glPushMatrix();
    glMultMatrixf(cam_pose_wc.inverse().transpose().cast<float>().eval().data());

    glBegin(GL_LINES);
    glColor3f(color.r, color.g, color.b);
    glLineWidth(color.size);
    drawFrustum(0.1f);
    glEnd();

    glPopMatrix();
  }


private:
  void draw_line(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) const
  {
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
  }

  void drawFrustum(const float w) const
  {
    const float h = w * 0.75f;
    const float z = w * 0.6f;
    // 四角錐の斜辺
    draw_line(0.0f, 0.0f, 0.0f, w, h, z);
    draw_line(0.0f, 0.0f, 0.0f, w, -h, z);
    draw_line(0.0f, 0.0f, 0.0f, -w, -h, z);
    draw_line(0.0f, 0.0f, 0.0f, -w, h, z);
    // 四角錐の底辺
    draw_line(w, h, z, w, -h, z);
    draw_line(-w, h, z, -w, -h, z);
    draw_line(-w, h, z, w, h, z);
    draw_line(-w, -h, z, w, -h, z);
  }
};