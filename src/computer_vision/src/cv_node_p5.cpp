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

#include <chrono>
#include <memory>
#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <image_transport/image_transport.hpp>
#include <image_geometry/pinhole_camera_model.h>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

enum {
  MAX_POINTS = 300
};

cv::Mat K = cv::Mat::zeros(3, 3, CV_64F); // Intrinsic parameters
cv::Mat T = cv::Mat::zeros(3, 4, CV_64F); // Extrinsic parameters

std::vector<cv::Point> points(MAX_POINTS);
long unsigned int np = 0; // Number of points

// Create mouse callback
void on_mouse(int event, int x, int y, int, void*)
{
  if(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
    if(np < points.size()) {
      points[np].x = x;
      points[np].y = y;
      np++;
    } else {
      std::cerr << "Can't store more points" << std::endl;
    }
  }
}

cv::Mat image_processing(const cv::Mat in_image, const cv::Mat depth_image);

cv::Point get_point(cv::Mat img, cv::Mat p3d, const int dist, const cv::Scalar color);
cv::Mat enclose_distance(cv::Mat img); // Enclose the image depending on the maximun distance and edit the image drawing the distance lines

cv::Mat get_mask(const cv::Mat img, const cv::Scalar lower, const cv::Scalar upper);
cv::Mat color_filter(const cv::Mat img); // Filtered the red and white lines of the floor

cv::Mat skeleton(const cv::Mat img);

void draw_points(cv::Mat img, const cv::Mat depth_img);

class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
   
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));

      sub_depth_ = this->create_subscription<sensor_msgs::msg::Image>( 
      "/head_front_camera/depth_registered/image_raw", qos, std::bind(&ComputerVisionSubscriber::depth_callback, this, std::placeholders::_1));

      sub_camera_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/head_front_camera/rgb/camera_info", qos, std::bind(&ComputerVisionSubscriber::camera_callback, this, std::placeholders::_1));
    
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);

      timer_ = create_wall_timer(std::chrono::seconds(1), std::bind(&ComputerVisionSubscriber::on_timer, this));
      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      // Create a window
      cv::namedWindow("P5");

      // Create mouse callback
      cv::setMouseCallback("P5", on_mouse, 0);

      // Create the 3 Trackbars and add to a window
      cv::createTrackbar("Option", "P5", nullptr, 2, 0);
      cv::createTrackbar("Iterations", "P5", nullptr, 100, 0);
      cv::createTrackbar("Distance", "P5", nullptr, 8, 0);
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw = cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw, depth_image_);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; 
      img_bridge.toImageMsg(out_image); 

      // Publish the data
      publisher_->publish(out_image);
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
      camera_model_.fromCameraInfo(msg);

      // Get intrinsic matrix K
      cv::Matx<double, 3, 3> kx;
      kx = camera_model_.intrinsicMatrix();

      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          K.at<double>(i, j) = kx(i, j);
        }
      }
    }

    void on_timer() const 
    {
      geometry_msgs::msg::TransformStamped t;

      // Look up for the transformation between the base of the robot and the camera
      try {
        t = tf_buffer_->lookupTransform(camera_model_.tfFrame(), "base_footprint", tf2::TimePointZero); 
      } catch (const tf2::TransformException & ex) {
        RCLCPP_INFO(this->get_logger(), "Could not transform from base_footprint %s: %s", camera_model_.tfFrame().c_str(), ex.what());  
        return;
      }

      // Build the matrix T with the translation and rotation
      Eigen::Vector3d v(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z);
      Eigen::Quaterniond q(t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z);
      Eigen::Matrix3d R = q.toRotationMatrix();

      for(int i = 0; i < 3; i++) {
        T.at<double>(i, 3) = v(i);
        for(int j = 0; j < 3; j++) {
          T.at<double>(i, j) = R(i, j);
        }
      }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_camera_;

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    image_geometry::PinholeCameraModel camera_model_;
    cv::Mat depth_image_;
};

cv::Point get_point(cv::Mat img, cv::Mat p3d, const int dist, const cv::Scalar color)
{
  cv::Mat p2d = cv::Mat::zeros(3, 1, CV_64F);

  p3d.at<double>(0, 0) = dist;
  p2d = K * T * p3d;

  cv::Point p(p2d.at<double>(0, 0) / p2d.at<double>(2, 0), p2d.at<double>(1, 0) / p2d.at<double>(2, 0));
  cv::circle(img, p, 5, color, -1);

  return p;
}

cv::Mat enclose_distance(cv::Mat img)
{
  cv::Mat dist_img;
  img.copyTo(dist_img);

  cv::Mat p3d = (cv::Mat_<double>(4, 1) << 0.0, 1.4, 0.0, 1.0);
  cv::Mat q3d = (cv::Mat_<double>(4, 1) << 0.0, -1.4, 0.0, 1.0);

  int dist = cv::getTrackbarPos("Distance", "P5");
  cv::Point p1, p2;

  for(int i = 3; i <= dist; i++) {
    cv::Scalar color = cv::Scalar(i * 20, i * 30, 255 - i * 15);
    p1 = get_point(img, p3d, i, color);
    p2 = get_point(img, q3d, i, color);
    cv::line(img, p1, p2, color, 3, cv::LINE_8); 
    cv::putText(img, std::to_string(i), cv::Point(p2.x + 7, p2.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
  }

  if(dist > 2) {
    // Points inside the width of 2.8 metres
    cv::rectangle(dist_img, cv::Point(0, 0), cv::Point(p1.x, dist_img.rows), cv::Scalar(0, 0, 0), -1, cv::LINE_8);
    cv::rectangle(dist_img, cv::Point(p2.x, 0), cv::Point(dist_img.cols, dist_img.rows), cv::Scalar(0, 0, 0), -1, cv::LINE_8);

    // Keep points further than the maximun distance
    cv::rectangle(dist_img, cv::Point(0, p1.y), cv::Point(dist_img.cols, dist_img.rows), cv::Scalar(0, 0, 0), -1, cv::LINE_8);
  }

  return dist_img;
}

cv::Mat get_mask(const cv::Mat img, const cv::Scalar lower, const cv::Scalar upper)
{
  // Convert to HSV
  cv::Mat hsv_img;
  cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

  // Structural elements
  cv::Mat element5 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::Mat element7 = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

  // Get mask
  cv::Mat mask;
  cv::inRange(hsv_img, lower, upper, mask);
  cv::erode(mask, mask, element5); // Remove the noise
  cv::dilate(mask, mask, element7); // Fill the holes

  return mask;
}

cv::Mat color_filter(const cv::Mat img)
{   
  // Get masks
  cv::Mat red_mask = get_mask(img, cv::Scalar(160, 70, 115), cv::Scalar(172, 255, 117));
  cv::Mat white_mask = get_mask(img, cv::Scalar(0, 0, 113), cv::Scalar(255, 255, 120));

  // Apply the masks
  cv::Mat mask = white_mask | red_mask;
  cv::Mat filter_img;
  img.copyTo(filter_img, mask);

  // Binarize the image
  for(int i = 0; i < filter_img.rows; i++) {
    for(int j = 0; j < filter_img.cols; j++ ) {
      cv::Vec3b p = filter_img.at<cv::Vec3b>(i,j);
      if(p[0] != 0 || p[1] != 0 || p[2] != 0) {
        filter_img.at<cv::Vec3b>(i, j)[1] = 255;
        filter_img.at<cv::Vec3b>(i, j)[2] = 255;
        filter_img.at<cv::Vec3b>(i, j)[0] = 255;
      } 
    }
  }

  return filter_img;
}

cv::Mat skeleton(const cv::Mat img)
{
  cv::Mat out_img, dist_img, filter_img;

  img.copyTo(out_img);
  dist_img = enclose_distance(out_img);
  filter_img = color_filter(dist_img);

  // Create the skeleton
  cv::Mat open, temp, skeleton = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Structural element
  int iterations = cv::getTrackbarPos("Iterations", "P5");

  for(int i = 0; i < iterations; i++) {
    cv::morphologyEx(filter_img, open, cv::MORPH_OPEN, element); // Openning
    cv::subtract(filter_img, open, temp);
    cv::erode(filter_img, filter_img, element); 
    cv::bitwise_or(skeleton, temp, skeleton);
  }

  // Draw skeleton
  for(int i = 0; i < out_img.rows; i++) {
    for(int j = 0; j < out_img.cols; j++ ) {
      cv::Vec3b p = skeleton.at<cv::Vec3b>(i,j);
      if(p[0] != 0 || p[1] != 0 || p[2] != 0) {
        out_img.at<cv::Vec3b>(i, j)[1] = 255;
      } 
    }
  }

  return out_img;
}

void draw_points(cv::Mat img, const cv::Mat depth_img)
{
  double x, y, z;
  std::string coor;

  for(long unsigned int i = 0; i < np; i++) {
    // Draw the point
    cv::circle(img, cv::Point(points[i].x, points[i].y), 5, cv::Scalar(255, 255, 255), -1);

    // Calculate 3D coordinates
    z = depth_img.at<float>(points[i].y, points[i].x); // filas, columnas
    x = (points[i].x - K.at<double>(0, 2)) * z / K.at<double>(0, 0);
    y = (points[i].y - K.at<double>(1, 2)) * z / K.at<double>(1, 1);

    // Write the coordinates
    std::ostringstream coor_stream;
    coor_stream << "[" << std::fixed << std::setprecision(2) << x << ", " << y << ", " << z << "]";
    cv::putText(img, coor_stream.str(), cv::Point(points[i].x + 10, points[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));
  }
}

cv::Mat image_processing(const cv::Mat in_image, const cv::Mat depth_image) 
{
  // Create output image
  cv::Mat out_image;
  
  // Processing
  out_image = in_image;

  // Get Trackbar’s value
  int key = cv::getTrackbarPos("Option", "P5");

  switch(key) {
    case 1:
      out_image = skeleton(out_image);
      draw_points(out_image, depth_image);
      break;
    case 2:
      depth_image.convertTo(out_image, CV_8U);
      cv::normalize(out_image, out_image, 0, 255, cv::NORM_MINMAX); // realazar
      draw_points(out_image, depth_image);
  }

  // Show image
  cv::imshow("P5", out_image);
  cv::waitKey(5);

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();

  return 0;
}
