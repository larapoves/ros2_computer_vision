/*
# Copyright (c) 2022 José Miguel Guerrero Hernández
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

/*
  Autora: Lara Poves
  Partes implementadas:
  - Detección de pelota en 2D y proyección 3D
  - Detección de pelota en 3D y proyección 2D
  - Proyección líneas

  - Funcionalidad extra:
    - Proyección de la pelota de 3D a 2D teniendo en cuenta el radio calculado en 3D.
    - Proyección de la pelota de 2D a 3D teniendo en cuenta el radio calculado en 2D.
*/

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <image_geometry/pinhole_camera_model.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

enum {
  MAX_BALLS = 75,
  MIN_POINTS = 100
};

bool is_person = false; // It's true if there is a person

std::vector<cv::Point3d> radius_img(MAX_BALLS);
std::vector<cv::Point3d> centers_img(MAX_BALLS); // Centers found in 2D convert to 3D
int nb_img = 0; // Number of balls

std::vector<cv::Mat> radius_cloud(MAX_BALLS); 
std::vector<cv::Mat> centers_cloud(MAX_BALLS); // Centers found in 3D
int nb_cloud = 0; // Number of balls

cv::Mat K = cv::Mat::zeros(3, 3, CV_32F); // Intrinsic parameters
cv::Mat T = cv::Mat::zeros(3, 4, CV_32F); // Extrinsic parameters

// Detect person
void person(const cv::Mat img);

// OpenCV functions
cv::Mat image_processing(const cv::Mat in_image, const cv::Mat depth_image);

// Main functions
cv::Mat balls_2D(const cv::Mat img, const cv::Mat depth_img, const bool draw);
cv::Mat distance_2D(const cv::Mat img);
cv::Mat draw_from3D(const cv::Mat img);

// Auxiliar functions
cv::Point get_point(cv::Mat img, const cv::Mat p3d, const bool camera, const bool draw, const cv::Scalar color, const int size);
cv::Mat filter_balls(const cv::Mat img);
cv::Mat binarize(const cv::Mat img);

// Point cloud functions
pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);

// Main functions
pcl::PointCloud<pcl::PointXYZRGB> detect_balls(pcl::PointCloud<pcl::PointXYZRGB> cloud, const bool draw);
pcl::PointCloud<pcl::PointXYZRGB> draw_centers2D(pcl::PointCloud<pcl::PointXYZRGB> cloud, const bool draw_circle);
pcl::PointCloud<pcl::PointXYZRGB> distance_3D(pcl::PointCloud<pcl::PointXYZRGB> cloud);

// Auxiliar functions
pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_filter(const pcl::PointCloud<pcl::PointXYZRGB> cloud);
pcl::PointCloud<pcl::PointXYZRGB> draw_cube(pcl::PointCloud<pcl::PointXYZRGB> cloud, const float side, const cv::Scalar color, const float x, const float y, const float z);

class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS(rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));

      sub_camera_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/head_front_camera/rgb/camera_info", qos, std::bind(&ComputerVisionSubscriber::camera_callback, this, std::placeholders::_1));

      sub_depth_ = this->create_subscription<sensor_msgs::msg::Image>( 
      "/head_front_camera/depth_registered/image_raw", qos, std::bind(&ComputerVisionSubscriber::depth_callback, this, std::placeholders::_1));
    
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);

      // Create a window
      cv::namedWindow("PRACTICA_FINAL");

      // Create the Trackbars
      cv::createTrackbar("Option", "PRACTICA_FINAL", nullptr, 2, 0);
      cv::createTrackbar("Distance", "PRACTICA_FINAL", nullptr, 8, 0);
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw, depth_image_);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; 
      img_bridge.toImageMsg(out_image); 

      // Publish the data
      publisher_ -> publish(out_image);
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    { 
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
      depth_image_ = cv_ptr->image;
    }

    void camera_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) 
    {     
      // Create PinholeCameraModel from CameraInfo message
      image_geometry::PinholeCameraModel camera_model;
      camera_model.fromCameraInfo(msg);

      // Get intrinsic matrix K
      cv::Matx<float, 3, 3> kx;
      kx = camera_model.intrinsicMatrix();

      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          K.at<float>(i, j) = kx(i, j);
        }
      }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_camera_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
    cv::Mat depth_image_;
};

class PCLSubscriber : public rclcpp::Node
{
  public:
    PCLSubscriber()
    : Node("pcl_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos, std::bind(&PCLSubscriber::topic_callback_3d, this, std::placeholders::_1));
    
      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pcl_points", qos);

      timer_ = create_wall_timer(std::chrono::seconds(1), std::bind(&PCLSubscriber::on_timer, this));
      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

  private:
    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud);
      
      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(pcl_pointcloud, output);
      output.header = msg->header;

      // Publish the data
      publisher_3d_ -> publish(output);
    }

    void on_timer() const 
    {
      geometry_msgs::msg::TransformStamped t;

      // Look up for the transformation between the base of the robot and the camera
      try {
        t = tf_buffer_->lookupTransform("head_front_camera_rgb_optical_frame", "base_footprint", tf2::TimePointZero); 
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO(this->get_logger(), "Could not transform from base_footprint to head_front_camera_rgb_optical_frame: %s", ex.what());  
        return;
      }

      // Build the matrix T with the translation and rotation
      Eigen::Vector3d v(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z);
      Eigen::Quaterniond q(t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z);
      Eigen::Matrix3d R = q.toRotationMatrix();

      for(int i = 0; i < 3; i++) {
        T.at<float>(i, 3) = v(i);
        for(int j = 0; j < 3; j++) {
          T.at<float>(i, j) = R(i, j);
        }
      }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_3d_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_3d_;

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
};

cv::Point get_point(cv::Mat img, const cv::Mat p3d, const bool camera, const bool draw, const cv::Scalar color = cv::Scalar(0), const int size = 1)
{
  cv::Mat p2d;

  // Convert from 3D to 2D
  if(camera){
    p2d = K * p3d;
  } else {
    p2d = K * T * p3d;
  }

  // Calculate the point in 2D
  cv::Point p(p2d.at<float>(0, 0) / p2d.at<float>(2, 0), p2d.at<float>(1, 0) / p2d.at<float>(2, 0));

  // Draw the point
  if (draw) {
    cv::circle(img, p, size, color, -1);
  }

  return p;
}

cv::Mat distance_2D(const cv::Mat img)
{
  cv::Mat dist_img;
  img.copyTo(dist_img);

  cv::Mat p3d = (cv::Mat_<float>(4, 1) << 0.0, 1.0, 0.0, 1.0);
  cv::Mat q3d = (cv::Mat_<float>(4, 1) << 0.0, -1.0, 0.0, 1.0);

  int dist = cv::getTrackbarPos("Distance", "PRACTICA_FINAL");
  cv::Point p1, p2;

  for(int i = 3; i <= dist; i++) {
    p3d.at<float>(0, 0) = i;
    q3d.at<float>(0, 0) = i;

    cv::Scalar color = cv::Scalar(0, (i - 3) * 51, 255 - (i - 3) * 51);
    p1 = get_point(dist_img, p3d, false, true, color, 5);
    p2 = get_point(dist_img, q3d, false, true, color, 5);

    cv::line(dist_img, p1, p2, color, 3, cv::LINE_8);
    cv::putText(dist_img, std::to_string(i), cv::Point(p2.x + 7, p2.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
  }

  return dist_img;
}

cv::Mat binarize(const cv::Mat img)
{
  cv::Mat gray, binary;

  // Convert to gray scale
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  binary = gray.clone();

  for(int i = 0; i < gray.rows; i++) {
    for(int j = 0; j < gray.cols; j++) {
      if(gray.at<uchar>(i, j) == 0) {
        binary.at<uchar>(i, j) = 0;
      } else {
        binary.at<uchar>(i, j) = 255;
      }
    }
  }

  return binary;
}

cv::Mat filter_balls(const cv::Mat img)
{
  // Filter pink balls
  cv::Mat mask, balls;
  cv::inRange(img, cv::Scalar(35, 0, 35), cv::Scalar(255, 60, 255), mask);
  img.copyTo(balls, mask);

  // Remove noise
  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Structural element
  cv::morphologyEx(balls, balls, cv::MORPH_OPEN, element); // Openning

  // Binarize
  cv::Mat binary = binarize(balls);
  return binary;
}

cv::Mat balls_2D(const cv::Mat img, const cv::Mat depth_img, const bool draw)
{
  cv::Mat binary_img = filter_balls(img);

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(binary_img, circles, cv::HOUGH_GRADIENT, 1, binary_img.rows / 16, 50, 15, 1, 250);

  // Copy the original image
  cv::Mat out_img;
  img.copyTo(out_img);

  // Draw 3D centers 
  if (draw) {
    for(int i = 0; i < nb_cloud; i++) {
      get_point(out_img, centers_cloud[i], true, true, cv::Scalar(128, 128, 128), 7);
    }
  }

  // Draw the circles and the centers from 2D
  nb_img = circles.size();
  for(size_t i = 0; i < circles.size(); i++) {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    int radius = c[2];

    // Draw the circle and the center
    if (draw) {
      cv::circle(out_img, center, 1, cv::Scalar(0, 0, 0), 3, cv::LINE_AA); // Circle center
      cv::circle(out_img, center, radius, cv::Scalar(0, 0, 255), 3, cv::LINE_AA); // Circle outline
    }

    // Calculate 3D coordinates
    centers_img[i].z = depth_img.at<float>(center.y, center.x);
    centers_img[i].x = (center.x - K.at<float>(0, 2)) * centers_img[i].z / K.at<float>(0, 0);
    centers_img[i].y = (center.y - K.at<float>(1, 2)) * centers_img[i].z / K.at<float>(1, 1);

    radius_img[i].z = centers_img[i].z;
    radius_img[i].x = (center.x + radius - K.at<float>(0, 2)) * centers_img[i].z / K.at<float>(0, 0);
    radius_img[i].y = centers_img[i].y;
  }
  
  return out_img;
}

cv::Mat draw_from3D(const cv::Mat img)
{
  // Copy the original image
  cv::Mat out_img;
  img.copyTo(out_img);

  // Draw 3D centers and circles
  cv::Point center3D, radius3D, v;

  for(int i = 0; i < nb_cloud; i++) {
    center3D = get_point(out_img, centers_cloud[i], true, true, cv::Scalar(255, 255, 255), 3);
    radius3D = get_point(out_img, radius_cloud[i], true, false);

    // Calculate radius
    v = center3D - radius3D;
    float radius = sqrt(v.x * v.x + v.y * v.y);

    // Draw the circle and the radius
    cv::circle(out_img, center3D, radius, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
    cv::line(out_img, center3D, radius3D, cv::Scalar(255, 255, 255), 1.5, cv::LINE_8);
    cv::putText(out_img, "R", cv::Point(radius3D.x - 20, radius3D.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1.5);
  }

  return out_img;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_cube(pcl::PointCloud<pcl::PointXYZRGB> cloud, const float side, const cv::Scalar color, const float x, const float y, const float z)
{
  float dist = 0.01;
  pcl::PointXYZRGB p;

  // Set color
  p.r = color[2];
  p.g = color[1];
  p.b = color[0];

  for(int i = 0; i <= side / dist / 2; i++) {
    for(int j = 0; j <= side / dist / 2; j++) {
      for(int k = 0; k <= side / dist / 2; k++) {
        // + + +
        p.x = x + dist * i;
        p.y = y + dist * j;
        p.z = z + dist * k;
        cloud.push_back(p);

        // + - +
        p.y = y - dist * j;
        cloud.push_back(p);

        // + - -
        p.z = z - dist * k;
        cloud.push_back(p);

        // - - -
        p.x = x - dist * i;
        cloud.push_back(p);

        // - - +
        p.z = z + dist * k;
        cloud.push_back(p);

        // - + +
        p.y = y + dist * j;
        cloud.push_back(p);

        // - + -
        p.z = z - dist * k;
        cloud.push_back(p);

        // + + -
        p.x = x + dist * i;
        cloud.push_back(p);
      }
    }
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_centers2D(pcl::PointCloud<pcl::PointXYZRGB> cloud, const bool draw_circle)
{
  float radius = 0.0;

  for(int i = 0; i < nb_img; i++) {
    if (draw_circle) {
      // Calculate radius
      radius = radius_img[i].x - centers_img[i].x;
      
      // Draw the circle
      int num_points = 200;
      for (int p = 0; p < num_points; p++) {
        pcl::PointXYZRGB point;
        double angle = 2.0 * M_PI * p / num_points;
        point.x = centers_img[i].x + radius * cos(angle);
        point.y = centers_img[i].y + radius * sin(angle);
        point.z = centers_img[i].z + radius;
        point.r = 0;
        point.g = 255;
        point.b = 0;
        cloud.push_back(point);
      }
    }

    // Draw the center
    cloud = draw_cube(cloud, 0.075, cv::Scalar(0, 0, 0), centers_img[i].x, centers_img[i].y, centers_img[i].z + radius);
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> distance_3D(pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
  cv::Mat gazebo = (cv::Mat_<float>(4, 1) << 0.0, 1.0, 0.0, 1.0);
  cv::Mat camera = cv::Mat::zeros(3, 1, CV_32F);

  int dist = cv::getTrackbarPos("Distance", "PRACTICA_FINAL");

  for(int s = 0; s < 2; s++) {
    for(int i = 3; i <= dist; i++) {
      // Get the coordinates
      gazebo.at<float>(0, 0) = i; 
      camera = T * gazebo;

      // Draw the square
      cv::Scalar color = cv::Scalar(0, (i - 3) * 51, 255 - (i - 3) * 51);
      cloud = draw_cube(cloud, 0.1, color, camera.at<float>(0, 0), camera.at<float>(1, 0), camera.at<float>(2, 0));
    }
    gazebo.at<float>(1, 0) = -1.0;
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_filter(const pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pink (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

  for (const auto& point: cloud) {
    if(point.r >= 35 && point.b >= 43 && point.g <= 38) {
      pink->push_back(point);
    }
  }

  // Remove outliers values if there are any balls
  if(pink->size() > MIN_POINTS) {
    sor.setInputCloud(pink);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);
  }
  
  return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZRGB> detect_balls(pcl::PointCloud<pcl::PointXYZRGB> cloud, const bool draw)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_balls (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_p (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  pcl::SACSegmentation<pcl::PointXYZRGB> seg; // Segmentation object
  pcl::ExtractIndices<pcl::PointXYZRGB> extract; // Filtering object

  cloud_balls = color_filter(cloud);
  if(cloud_balls->size() <= MIN_POINTS) {
    return cloud;
  }

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_SPHERE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(0.01);

  nb_cloud = 0;
  while(cloud_balls->size() > MIN_POINTS)
  {
    // Segment the largest sphere component from the remaining cloud
    seg.setInputCloud(cloud_balls);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.size () == 0) {
      std::cerr << "Could not estimate a sphere model for the given dataset." << std::endl;
      break;
    }

    // Draw a cube in the center of the sphere
    if (draw) {
      cloud = draw_cube(cloud, 0.1, cv::Scalar(255, 0, 0), coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    }

    // Keep centers and radius in global variables
    centers_cloud[nb_cloud] = (cv::Mat_<float>(3, 1) << coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    radius_cloud[nb_cloud] = (cv::Mat_<float>(3, 1) << coefficients->values[0] + coefficients->values[3], coefficients->values[1], coefficients->values[2]);
    nb_cloud++;

    // Extract the inliers
    extract.setInputCloud(cloud_balls);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_p);

    // Create the filtering object
    extract.setNegative(true);
    extract.filter(*cloud_f);
    cloud_balls.swap(cloud_f);
  }

  return cloud;
}

void person(const cv::Mat img)
{
  cv::HOGDescriptor hog;
  std::vector<cv::Rect> rects;

  hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  hog.detectMultiScale(img, rects, 0, cv::Size(8, 8), cv::Size(20, 20), 1.05, 0.3, false);

  is_person = rects.size() > 0;
}

cv::Mat image_processing(const cv::Mat in_image, const cv::Mat depth_image) 
{
  // Processing
  cv::Mat out_image = in_image;

  // Get Trackbar’s value
  int key = cv::getTrackbarPos("Option", "PRACTICA_FINAL");

  switch(key) {
    case 1:
      person(out_image); // Update the global variable

      if(is_person) {
        out_image = balls_2D(out_image, depth_image, true);
        out_image = distance_2D(out_image);
      }
      break;
    
    case 2:
      out_image = balls_2D(out_image, depth_image, false);
      out_image = draw_from3D(out_image);
  }

  // Show image
  cv::imshow("PRACTICA_FINAL", out_image);
  cv::waitKey(10);

  return out_image;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_pointcloud = in_pointcloud;

  // Get Trackbar’s value
  int key = cv::getTrackbarPos("Option", "PRACTICA_FINAL");

  switch(key) {
    case 1:
      if(is_person) {
        out_pointcloud = detect_balls(out_pointcloud, true);
        out_pointcloud = draw_centers2D(out_pointcloud, false);
        out_pointcloud = distance_3D(out_pointcloud);
      }
      break;

    case 2:
      out_pointcloud = detect_balls(out_pointcloud, false);
      out_pointcloud = draw_centers2D(out_pointcloud, true);
    }

  return out_pointcloud;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::SingleThreadedExecutor exec;

  auto cv_node = std::make_shared<ComputerVisionSubscriber>();
  auto pcl_node = std::make_shared<PCLSubscriber>();
  exec.add_node(cv_node);
  exec.add_node(pcl_node);
  exec.spin();
  
  rclcpp::shutdown();
  return 0;
}