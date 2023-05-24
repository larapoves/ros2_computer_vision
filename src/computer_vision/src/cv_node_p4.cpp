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

cv::Mat image_processing(const cv::Mat in_image);

cv::Mat filter_color(const cv::Mat img, const cv::Scalar lower, const cv::Scalar upper);
cv::Mat binarize(const cv::Mat img);
cv::Mat drawing_contours(const cv::Mat img, const std::vector<std::vector<cv::Point>> contours, const std::vector<cv::Vec4i> hierarchy);

cv::Mat lines(const cv::Mat img);
cv::Mat balls(const cv::Mat img);
cv::Mat contours(const cv::Mat img);

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
    
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);

      // Create a window
      cv::namedWindow("P4");

      // Create the 3 Trackbars and add to a window
      cv::createTrackbar("O: Original; 1: Lines; 2: Balls; 3: Contours", "P4", nullptr, 3, 0);
      cv::createTrackbar("Hough accumulator", "P4", nullptr, 200, 0);
      cv::createTrackbar("Area", "P4", nullptr, 1000, 0);

      // Set Trackbar’s initial value
      cv::setTrackbarPos("Hough accumulator", "P4", 100);
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; 
      img_bridge.toImageMsg(out_image); 

      // Publish the data
      publisher_->publish(out_image);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

cv::Mat filter_color(const cv::Mat img, const cv::Scalar lower, const cv::Scalar upper)
{
  cv::Mat color_img, hsv_img, color_mask;

  // Convert to HSV
  cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

  // Apply the color mask 
  cv::inRange(hsv_img, lower, upper, color_mask);
  img.copyTo(color_img, color_mask);

  return color_img;
}

cv::Mat lines(const cv::Mat img)
{
  // Filter color green
  cv::Mat green_img = filter_color(img, cv::Scalar(40, 40, 40), cv::Scalar(70, 255, 255));

  // Convert to gray scale
  cv::cvtColor(green_img, green_img, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(green_img, green_img, cv::Size(3, 3), 0);

  // Edge detection
  cv::Canny(green_img, green_img, 75, 150, 3); // 3 es tamaño kernel

  // Get Hough accumulator value
  int value = cv::getTrackbarPos("Hough accumulator", "P4");

  // Standard Hough Line Transform
  std::vector<cv::Vec2f> lines; 
  cv::HoughLines(green_img, lines, 1, CV_PI / 180, value, 0, 0); // value -> umbral a partir del cual se considera detectada

  // Copy original image
  cv::Mat out_img;
  img.copyTo(out_img);

  // Draw the lines
  for(size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    cv::Point pt1, pt2;
    double a = std::cos(theta), b = std::sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * ( a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * ( a));
    cv::line(out_img, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  }

  return out_img;
}

cv::Mat binarize(const cv::Mat img)
{
  cv::Mat binary = img.clone();

  for(int i = 0; i < img.rows; i++) {
    for(int j = 0; j < img.cols; j++) {
      if(img.at<uchar>(i, j) == 0) {
        binary.at<uchar>(i, j) = 0;
      } else {
        binary.at<uchar>(i, j) = 255;
      }
    }
  }

  return binary;
}

cv::Mat balls(const cv::Mat img)
{
  // Filter color blue
  cv::Mat blue_img = filter_color(img, cv::Scalar(80, 20, 20), cv::Scalar(150, 255, 255));

  // Convert to gray scale
  cv::cvtColor(blue_img, blue_img, cv::COLOR_BGR2GRAY);

  // Binarize
  cv::Mat binary_img = binarize(blue_img);

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(binary_img, circles, cv::HOUGH_GRADIENT, 1, binary_img.rows / 16, 50, 15, 1, 250);
  // umbral de dtección, un valor más pequeño permite detectar circulos más facilmente
  // umbral acumulativo, a más pequeño detección más estricta
  // radio mínimo y máximo del circulo

  // Copy original image
  cv::Mat out_img;
  img.copyTo(out_img);

  // Draw the circles and the centers
  for(size_t i = 0; i < circles.size(); i++) {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    int radius = c[2];
    cv::circle(out_img, center, 1, cv::Scalar(0, 255, 0), 3, cv::LINE_AA); // Circle center
    cv::circle(out_img, center, radius, cv::Scalar(255, 0,255), 3, cv::LINE_AA); // Circle outline
  }
  
  return out_img;
}

cv::Mat drawing_contours(const cv::Mat img, const std::vector<std::vector<cv::Point>> contours, const std::vector<cv::Vec4i> hierarchy)
{
  // Copy original image
  cv::Mat drawing;
  img.copyTo(drawing);

  srand(0);
  double value = cv::getTrackbarPos("Area", "P4");
  std::vector<cv::Moments> mu(contours.size());

  for(size_t i = 0; i < contours.size(); i++) {
    double area = contourArea(contours[i]);

    if(area > value) { 
      int b = rand() & 255, g = rand() & 255, r = rand() & 255;

      // Drawing contours
      cv::drawContours(drawing, contours, i, cv::Scalar(b, g, r), 2, cv::LINE_8, hierarchy, 1);

      // Draw centroid
      mu[i] = cv::moments(contours[i]);
      int cx = mu[i].m10 / mu[i].m00;
      int cy = mu[i].m01 / mu[i].m00;
      cv::circle(drawing, cv::Point(cx, cy), 4, cv::Scalar(b, g, r), -1);

      // Write area
      cv::putText(drawing, std::to_string(area), cv::Point(cx + 5, cy - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(b, g, r), 2); 
    }
  }

  return drawing;
}

cv::Mat contours(const cv::Mat img)
{
  // Filter color green
  cv::Mat green_img = filter_color(img, cv::Scalar(40, 40, 40), cv::Scalar(70, 255, 255));

  // Filter color blue
  cv::Mat blue_img = filter_color(img, cv::Scalar(80, 50, 20), cv::Scalar(150, 255, 255));

  // Merge the two filtered images
  cv::Mat filter_img(img.rows, img.cols, CV_8UC3);
  cv::addWeighted(green_img, 1, blue_img, 1, 0.0, filter_img);

  // Convert to gray scale
  cv::cvtColor(filter_img, filter_img, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(filter_img, filter_img, cv::Size(3, 3), 0); // Avoid false contours in the intersection of the line and the ball

  // Binarize
  cv::Mat binary_img = binarize(filter_img);

  // In order to detect them better
  cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Structural element
  cv::morphologyEx(binary_img, binary_img, cv::MORPH_OPEN, element); // Openning
  cv::GaussianBlur(binary_img, binary_img, cv::Size(5, 5), 0); // Join the line and the balls

  // Add a black edge to the image
  int border = 2;
  cv::Mat border_img = cv::Mat::zeros(cv::Size(binary_img.cols + 2 * border, binary_img.rows + 2 * border), binary_img.type());
  binary_img.copyTo(border_img(cv::Rect(border, border, binary_img.cols, binary_img.rows)));

  // Extract edges
  cv::Mat edges;
  cv::Canny(border_img, edges, 25, 50, 3); 
  
  // Contours
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  // Relocate the contours
  for(size_t i = 0; i < contours.size(); i++) {
    for(size_t j = 0; j < contours[i].size(); j++) {
      contours[i][j].x -= border;
      contours[i][j].y -= border;
    }
  }

  cv::Mat drawing = drawing_contours(img, contours, hierarchy);

  return drawing;
}

cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  
  // Processing
  out_image = in_image;

  // Get Trackbar’s value
  int key = cv::getTrackbarPos("O: Original; 1: Lines; 2: Balls; 3: Contours", "P4");

  switch(key) {
    case 1:
      out_image = lines(out_image);
      break;
    case 2:
      out_image = balls(out_image);
      break;
    case 3:
      out_image = contours(out_image);
  }

  // Show image in a different window
  cv::imshow("P4", out_image);
  cv::waitKey(1);

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();

  return 0;
}