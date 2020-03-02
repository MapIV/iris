#include "pangolin_viewer.hpp"

namespace vllm
{
PangolinViewer::PangolinViewer(const std::shared_ptr<System>& system_ptr)
    : system_ptr(system_ptr), s_cam(makeCamera()), handler(pangolin::Handler3D(s_cam))
{
  // setup Pangolin viewer
  pangolin::CreateWindowAndBind("VLLM", 1024, 768);
  glEnable(GL_DEPTH_TEST);

  // Ensure that blending is enabled for rendering text.
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  d_cam = (pangolin::CreateDisplay()
               .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
               .SetHandler(&handler));

  // setup GUI
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));
  gui_quit = std::make_shared<pangolin::Var<bool>>("ui.Quit", false, false);
  gui_raw_camera = std::make_shared<pangolin::Var<bool>>("ui.raw_camera", false, true);
  gui_source_normals = std::make_shared<pangolin::Var<bool>>("ui.source_normals", false, true);
  gui_target_normals = std::make_shared<pangolin::Var<bool>>("ui.target_normals", false, true);
  Eigen::Vector2d gain = system_ptr->getGain();
  gui_scale_gain = std::make_shared<pangolin::Var<double>>("ui.scale_gain", gain(0), 0.0, 50.0);
  gui_pitch_gain = std::make_shared<pangolin::Var<double>>("ui.pitch_gain", gain(1), 0.0, 50.0);
  // gui_gpd = std::make_shared<pangolin::Var<bool>>("ui.global_distribution", false, true);
}

pangolin::OpenGlRenderState PangolinViewer::makeCamera(
    const Eigen::Vector3f& from,
    const Eigen::Vector3f& to,
    const pangolin::AxisDirection up)
{
  return pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(
          640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(
          from.x(), from.y(), from.z(), to.x(), to.y(), to.z(), up));
}

void PangolinViewer::drawString(const std::string& str, const Color& color) const
{
  glColor3f(color.r, color.g, color.b);
  glPointSize(color.size);
  glLineWidth(color.size);
  pangolin::GlFont::I().Text(str).DrawWindow(200, 50 - 2.0f * pangolin::GlFont::I().Height());
}

void PangolinViewer::drawGPD(const GPD& gpd) const
{
  const size_t N = gpd.size();
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        const LPD& lpd = gpd.at(i, j, k);
        if (lpd.N < 20) continue;
        glPushMatrix();
        glMultMatrixf(lpd.T.transpose().eval().data());

        drawRectangular(lpd.sigma.x(), lpd.sigma.y(), lpd.sigma.z());

        glPopMatrix();
      }
    }
  }
}

void PangolinViewer::drawTrajectory(const std::vector<Eigen::Vector3f>& trajectory, const Color& color)
{
  glBegin(GL_LINE_STRIP);
  glLineWidth(color.size);
  int i = 0;
  for (const Eigen::Vector3f& v : trajectory) {
    glColor3fv(convertRGB(Eigen::Vector3f(static_cast<float>(i % 360), 1.f, 1.f)).data());
    glVertex3f(v.x(), v.y(), v.z());
    i += 2;
  }
  glEnd();
}


void PangolinViewer::drawPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Color& color) const
{
  glColor3f(color.r, color.g, color.b);
  glPointSize(color.size);

  PangolinCloud pc(cloud);
  pc.drawPoints();
}

void PangolinViewer::drawGridLine() const
{
  Eigen::Matrix4f origin = Eigen::Matrix4f::Identity();

  glPushMatrix();
  glMultTransposeMatrixf(origin.data());

  glLineWidth(1);
  glColor3f(0.3f, 0.3f, 0.3f);

  glBegin(GL_LINES);
  constexpr float interval_ratio = 0.1f;
  constexpr float grid_min = -100.0f * interval_ratio;
  constexpr float grid_max = 100.0f * interval_ratio;
  for (float x = -10.f; x <= 10.f; x += 1.0f) {
    drawLine(x * 10.0f * interval_ratio, grid_min, 0, x * 10.0f * interval_ratio, grid_max, 0);
  }
  for (float y = -10.f; y <= 10.f; y += 1.0f) {
    drawLine(grid_min, y * 10.0f * interval_ratio, 0, grid_max, y * 10.0f * interval_ratio, 0);
  }
  glEnd();
  glPopMatrix();

  // coordinate axis
  glLineWidth(3.0f);
  glBegin(GL_LINES);
  constexpr float axis = 1.0f;
  glColor3f(1.0f, 0.0f, 0.0f);
  drawLine(0, 0, 0, axis, 0, 0);
  glColor3f(0.0f, 1.0f, 0.0f);
  drawLine(0, 0, 0, 0, axis, 0);
  glColor3f(0.0f, 0.0f, 1.0f);
  drawLine(0, 0, 0, 0, 0, axis);
  glEnd();
}

void PangolinViewer::drawCamera(const Eigen::Matrix4f& cam_pose, const Color& color) const
{
  glPushMatrix();
  glMultMatrixf(cam_pose.transpose().eval().data());

  glBegin(GL_LINES);
  glColor3f(color.r, color.g, color.b);
  glLineWidth(color.size);
  drawFrustum(0.1f);
  glEnd();

  glPopMatrix();
}

void PangolinViewer::drawLine(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) const
{
  glVertex3f(x1, y1, z1);
  glVertex3f(x2, y2, z2);
}

void PangolinViewer::drawNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const Color& color,
    int skip) const
{
  glBegin(GL_LINES);
  glColor4f(color.r, color.g, color.b, 0.4f);
  glLineWidth(color.size);
  for (size_t i = 0; i < cloud->size(); i += skip) {
    Eigen::Vector3f p = cloud->at(i).getArray3fMap();
    Eigen::Vector3f n = normals->at(i).getNormalVector3fMap();
    n = 0.2f * n;  // 200mm
    if (std::isfinite(n.x()))
      drawLine(p.x(), p.y(), p.z(), p.x() + n.x(), p.y() + n.y(), p.z() + n.z());
  }
  glEnd();
}

void PangolinViewer::drawCorrespondences(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const pcl::CorrespondencesPtr& correspondences,
    const Color& color) const
{
  glBegin(GL_LINES);
  glColor4f(color.r, color.g, color.b, 0.4f);
  glLineWidth(color.size);
  for (const pcl::Correspondence& c : *correspondences) {
    pcl::PointXYZ p1 = source->at(c.index_query);
    pcl::PointXYZ p2 = target->at(c.index_match);
    drawLine(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
  }
  glEnd();
}

void PangolinViewer::drawRectangular(const float x, const float y, const float z) const
{
  const GLfloat x1 = x / 2.0f;
  const GLfloat x2 = -x1;
  const GLfloat y1 = y / 2.0f;
  const GLfloat y2 = -y1;
  const GLfloat z1 = z / 2.0f;
  const GLfloat z2 = -z1;

  const GLfloat verts[] = {
      x1, y1, z1, x2, y1, z1, x1, y2, z1, x2, y2, z1,  //
      x1, y1, z1, x2, y1, z1, x1, y1, z2, x2, y1, z2,  //
      x1, y1, z1, x1, y2, z1, x1, y1, z2, x1, y2, z2,  //
      x2, y2, z2, x1, y2, z2, x2, y1, z2, x1, y1, z2,  //
      x2, y2, z2, x1, y2, z2, x2, y2, z1, x1, y2, z1,  //
      x2, y2, z2, x2, y1, z2, x2, y2, z1, x2, y1, z1,  //
  };

  glVertexPointer(3, GL_FLOAT, 0, verts);
  glEnableClientState(GL_VERTEX_ARRAY);

  glColor4f(1.0f, 0.0f, 0.0f, 0.1f);
  glDrawArrays(GL_TRIANGLE_STRIP, 8, 4);
  glDrawArrays(GL_TRIANGLE_STRIP, 20, 4);

  glColor4f(0.0f, 1.0f, 0.0f, 0.1f);
  glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);
  glDrawArrays(GL_TRIANGLE_STRIP, 16, 4);

  glColor4f(0.0f, 0.0f, 1.0f, 0.1f);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glDrawArrays(GL_TRIANGLE_STRIP, 12, 4);

  glDisableClientState(GL_VERTEX_ARRAY);
}


void PangolinViewer::drawFrustum(const float w) const
{
  const float h = w * 0.75f;
  const float z = w * 0.6f;

  // hypotenuse of frustum
  drawLine(0.0f, 0.0f, 0.0f, w, h, z);
  drawLine(0.0f, 0.0f, 0.0f, w, -h, z);
  drawLine(0.0f, 0.0f, 0.0f, -w, -h, z);
  drawLine(0.0f, 0.0f, 0.0f, -w, h, z);
  // bottom of frustum
  drawLine(w, h, z, w, -h, z);
  drawLine(-w, h, z, -w, -h, z);
  drawLine(-w, h, z, w, h, z);
  drawLine(-w, -h, z, w, -h, z);
}

Eigen::Vector3f PangolinViewer::convertRGB(Eigen::Vector3f hsv)
{
  const float max = hsv(2);
  const float min = max * (1 - hsv(1));
  const float H = hsv(0);
  const float D = max - min;
  if (H < 60) return {max, H / 60 * D + min, min};
  if (H < 120) return {(120 - H) / 60 * D + min, max, min};
  if (H < 180) return {min, max, (H - 120) / 60 * D + min};
  if (H < 240) return {min, (240 - H) / 60 * D + min, max};
  if (H < 300) return {(H - 240) / 60 * D + min, min, max};
  if (H < 360) return {max, min, (360 - H) / 60 * D + min};
  return {255, 255, 255};
}

}  // namespace vllm