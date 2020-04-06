#include "viewer/pangolin_viewer.hpp"

namespace vllm
{
namespace viewer
{
PangolinViewer::PangolinViewer(const std::shared_ptr<System>& system_ptr)
    : system_ptr(system_ptr), loop_flag(true) {}

void PangolinViewer::clear()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  d_cam.Activate(*s_cam);
}

void PangolinViewer::swap()
{
  pangolin::FinishFrame();
}

void PangolinViewer::setIMU(const std::vector<Eigen::Matrix4f>& pose)
{
  std::lock_guard lock(imu_mtx);
  imu_poses = pose;
}

void PangolinViewer::init()
{
  s_cam = std::make_shared<pangolin::OpenGlRenderState>(makeCamera());
  handler = std::make_shared<pangolin::Handler3D>(pangolin::Handler3D(*s_cam));

  // setup Pangolin viewer
  pangolin::CreateWindowAndBind("VLLM", 1424, 968);
  glEnable(GL_DEPTH_TEST);

  // Ensure that blending is enabled for rendering text.
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  d_cam = (pangolin::CreateDisplay()
               .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
               .SetHandler(&(*handler)));

  // setup GUI
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));
  gui_quit = std::make_shared<pangolin::Var<bool>>("ui.Quit", false, false);
  gui_reset = std::make_shared<pangolin::Var<bool>>("ui.Reset", false, false);
  gui_vslam_camera = std::make_shared<pangolin::Var<bool>>("ui.vslam_camera", true, true);
  gui_source_normals = std::make_shared<pangolin::Var<bool>>("ui.source_normals", false, true);
  gui_target_normals = std::make_shared<pangolin::Var<bool>>("ui.target_normals", false, true);
  gui_target_normals = std::make_shared<pangolin::Var<bool>>("ui.target_normals", false, true);
  gui_correspondences = std::make_shared<pangolin::Var<bool>>("ui.correspondences", true, true);
  gui_imu = std::make_shared<pangolin::Var<bool>>("ui.IMU", true, true);

  // Initialize local map
  target_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  target_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
  localmap_info = system_ptr->getMap()->getLocalmapInfo();
  *target_cloud = *system_ptr->getMap()->getTargetCloud();
  *target_normals = *system_ptr->getMap()->getTargetNormals();

  colored_target_cloud = colorizePointCloud(target_cloud);
  target_normals_color = colorizeNormals(target_normals);

  const optimize::Gain& optimize_gain = system_ptr->getOptimizeGain();
  gui_scale_gain = std::make_shared<pangolin::Var<float>>("ui.scale_gain", optimize_gain.scale, 0.0f, 50.0f);
  gui_smooth_gain = std::make_shared<pangolin::Var<float>>("ui.smooth_gain", optimize_gain.smooth, 0.0f, 50.0f);
  gui_latitude_gain = std::make_shared<pangolin::Var<float>>("ui.latitude_gain", optimize_gain.latitude, 0.0f, 50.0f);
  gui_altitude_gain = std::make_shared<pangolin::Var<float>>("ui.altitude_gain", optimize_gain.altitude, 0.0f, 50.0f);

  gui_recollection = std::make_shared<pangolin::Var<unsigned int>>("ui.recollection", system_ptr->getRecollection(), 0, 200);

  // Eigen::Vector2d distance = system_ptr->getSearchDistance();
  // unsigned int recollect = system_ptr->getRecollection();
  // gui_distance_min = std::make_shared<pangolin::Var<double>>("ui.distance_min", distance(0), 0.0, 1.0);
  // gui_distance_max = std::make_shared<pangolin::Var<double>>("ui.distance_max", distance(1), 0.0, 3.0);
}

void PangolinViewer::loop()
{
  init();
  while (loop_flag.load()) {
    execute();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
}

void PangolinViewer::startLoop()
{
  if (system_ptr == nullptr) {
    std::cout << "syste_ptr is nullptr" << std::endl;
    exit(EXIT_FAILURE);
  }
  viewer_thread = std::thread(&PangolinViewer::loop, this);
}

void PangolinViewer::quitLoop()
{
  loop_flag.store(false);
  if (viewer_thread.joinable())
    viewer_thread.join();
}

void PangolinViewer::execute()
{
  clear();

  drawGridLine();
  drawString("VLLM", {1.0f, 1.0f, 0.0f, 3.0f});

  system_ptr->popPublication(publication);

  map::Info new_localmap_info = system_ptr->getMap()->getLocalmapInfo();
  if (new_localmap_info != localmap_info) {
    localmap_info = new_localmap_info;
    *target_cloud = *system_ptr->getMap()->getTargetCloud();
    *target_normals = *system_ptr->getMap()->getTargetNormals();
    colored_target_cloud = colorizePointCloud(target_cloud);
    target_normals_color = colorizeNormals(target_normals);
    return;
  }
  {
    std::lock_guard lock(imu_mtx);
    drawPoses(imu_poses);
  }


  drawPointCloud(colored_target_cloud, {0.6f, 0.6f, 0.6f, 1.0f});
  if (*gui_target_normals)
    drawNormals(target_cloud, target_normals, target_normals_color, 3);

  drawPointCloud(publication.cloud, {1.0f, 1.0f, 0.0f, 2.0f});
  if (*gui_source_normals)
    drawNormals(publication.cloud, publication.normals, {1.0f, 1.0f, 1.0f, 1.0f});

  if (*gui_vslam_camera) {
    drawCamera(publication.offset_camera, {1.0f, 0.0f, 1.0f, 1.0f});
    drawTrajectory(publication.offset_trajectory, false, {1.0f, 0.0f, 1.0f, 1.0f});
  }

  if (*gui_correspondences) {
    if (publication.localmap_info == localmap_info)
      drawCorrespondences(publication.cloud, target_cloud, publication.correspondences, {0.0f, 0.0f, 1.0f, 2.0f});
  }

  drawCamera(publication.vllm_camera, {1.0f, 0.0f, 0.0f, 1.0f});
  drawTrajectory(publication.vllm_trajectory, true);

  if (gui_scale_gain->GuiChanged() || gui_smooth_gain->GuiChanged() || gui_latitude_gain->GuiChanged() || gui_altitude_gain->GuiChanged())
    system_ptr->setOptimizeGain({*gui_scale_gain, *gui_smooth_gain, *gui_latitude_gain, *gui_altitude_gain});

  if (gui_recollection->GuiChanged())
    system_ptr->setRecollection(*gui_recollection);

  // Eigen::Vector2d distance(*gui_distance_min, *gui_distance_max);
  // system_ptr->setSearchDistance(distance);

  swap();

  // if (pangolin::Pushed(*gui_quit))
  //   return -1;

  imu_use_flag.store(*gui_imu);

  if (pangolin::Pushed(*gui_reset))
    system_ptr->requestReset();
}

pangolin::OpenGlRenderState PangolinViewer::makeCamera(
    const Eigen::Vector3f& from,
    const Eigen::Vector3f& to,
    const pangolin::AxisDirection up)
{
  return pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(
          640, 480, 420, 420, 320, 240, 0.2, 200),
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

void PangolinViewer::drawPoses(const std::vector<Eigen::Matrix4f>& poses) const
{
  glBegin(GL_LINES);
  glLineWidth(1.0);
  int c = 0;
  for (int i = 0; i < poses.size(); i += 10) {
    const Eigen::Matrix4f& pose = poses.at(i);
    glColor3fv(convertRGB(Eigen::Vector3f(static_cast<float>(c++ % 360), 1.f, 1.f)).data());

    Eigen::Matrix3f R = normalizeRotation(pose.topLeftCorner(3, 3));

    Eigen::Vector3f t = pose.topRightCorner(3, 1);
    Eigen::Vector3f f = t + 0.3 * R * Eigen::Vector3f::UnitZ();
    drawLine(t.x(), t.y(), t.z(), f.x(), f.y(), f.z());
  }
  glEnd();
}


void PangolinViewer::drawTrajectory(const std::vector<Eigen::Vector3f>& trajectory, bool colorful, const Color& color)
{
  glBegin(GL_LINE_STRIP);
  glLineWidth(color.size);
  glColor3f(color.r, color.g, color.b);

  int i = 0;
  for (const Eigen::Vector3f& v : trajectory) {
    if (colorful)
      glColor3fv(convertRGB(Eigen::Vector3f(static_cast<float>(i++ % 360), 1.f, 1.f)).data());
    glVertex3f(v.x(), v.y(), v.z());
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

void PangolinViewer::drawPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, const Color& color) const
{
  // glColor3f(color.r, color.g, color.b);
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
  constexpr float max = 100;
  constexpr float interval = 2.0f;

  constexpr float grid_min = -max * interval;
  constexpr float grid_max = max * interval;
  for (float x = -max; x <= max; x += interval) {
    drawLine(x * interval, grid_min, 0, x * interval, grid_max, 0);
  }
  for (float y = -max; y <= max; y += interval) {
    drawLine(grid_min, y * interval, 0, grid_max, y * interval, 0);
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
  glMultMatrixf(cam_pose.eval().data());

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
    n = 0.4f * n;
    if (std::isfinite(n.x()))
      drawLine(p.x(), p.y(), p.z(), p.x() + n.x(), p.y() + n.y(), p.z() + n.z());
  }
  glEnd();
}
void PangolinViewer::drawNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const std::vector<Color>& colors,
    int skip) const
{
  glBegin(GL_LINES);
  glLineWidth(1.0);
  for (size_t i = 0; i < cloud->size(); i += skip) {
    Eigen::Vector3f p = cloud->at(i).getArray3fMap();
    Eigen::Vector3f n = normals->at(i).getNormalVector3fMap();
    Color c = colors.at(i);

    glColor4f(c.r, c.g, c.b, 0.4f);

    n = 0.2f * n;
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
  glColor4f(color.r, color.g, color.b, 0.9f);
  glLineWidth(color.size);
  for (const pcl::Correspondence& c : *correspondences) {
    if (static_cast<size_t>(c.index_query) >= source->size() || static_cast<size_t>(c.index_match) >= target->size())
      break;
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

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PangolinViewer::colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGBA>);
  colored->reserve(cloud->size());

  const uint8_t MAX = 150;
  for (const pcl::PointXYZ& p : *cloud) {
    uint8_t tmp = static_cast<uint8_t>(std::min(std::abs(20 * p.z), 150.0f));
    pcl::PointXYZRGBA c;
    c.r = c.g = MAX;
    c.b = tmp;
    c.a = static_cast<uint8_t>(255 - tmp);
    c.x = p.x;
    c.y = p.y;
    c.z = p.z;
    colored->push_back(c);
  }
  return colored;
}

std::vector<Color> PangolinViewer::colorizeNormals(const pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
  std::vector<Color> colors;
  colors.reserve(normals->size());

  constexpr float max = 1.0f;
  constexpr float min = 0.1f;
  constexpr float gain = (max - min);
  for (const pcl::Normal& n : *normals) {
    Color c;
    c.r = gain * std::abs(n.normal_x) + min;
    c.g = gain * std::abs(n.normal_y) + min;
    c.b = gain * std::abs(n.normal_z) + min;
    colors.push_back(c);
  }
  return colors;
}

}  // namespace viewer
}  // namespace vllm