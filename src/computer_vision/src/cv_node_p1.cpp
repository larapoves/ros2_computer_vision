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
#include <math.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

enum {
  OP1 = 49,
  OP2,
  OP3,
  OP4,
  OP5,
  OP6
};

cv::Mat image_processing(const cv::Mat in_image, int key);

float get_h(float r, float g, float b);

void RGB2CMY(cv::Mat out_image);
void RGB2HSV(cv::Mat out_image);
void RGB2HSI(cv::Mat out_image);
void RGB2HSI_openCV(cv::Mat out_image, cv::Mat in_image);

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

      key_ = OP1; // Original image
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw = cv_ptr->image;

      // Read key
      int key = cv::waitKey(1); // waiting time for a key pressed
      if(key >= OP1 && key <= OP6) {
        key_ = key;
      }

      // Image processing
      cv::Mat cv_image = image_processing(image_raw, key_);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

      // Publish the data
      publisher_ -> publish(out_image);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    int key_;
};

float get_h(float r, float g, float b)
{
  float h;

  h = 360 / (2 * M_PI) * std::acos((0.5 * ((r - g) + (r - b))) / std::sqrt(std::pow(r - b, 2) + (r - b) * (g - b)));
  if(b > g) {
    h = 360 - h;
  }
  h = h / (360) * 255;

  return h;
}

void RGB2CMY(cv::Mat out_image)
{
  for(int i = 0; i < out_image.rows; i++) {
    for(int j = 0; j < out_image.cols; j++ ) { 
      out_image.at<cv::Vec3b>(i,j)[0] = 255 - out_image.at<cv::Vec3b>(i,j)[0];
      out_image.at<cv::Vec3b>(i,j)[1] = 255 - out_image.at<cv::Vec3b>(i,j)[1];
      out_image.at<cv::Vec3b>(i,j)[2] = 255 - out_image.at<cv::Vec3b>(i,j)[2];
    }
  }
}

void RGB2HSV(cv::Mat out_image)
{
  float r, g, b;

  for(int i = 0; i < out_image.rows; i++) {
    for(int j = 0; j < out_image.cols; j++ ) { 
      b = out_image.at<cv::Vec3b>(i,j)[0] / 255.0; 
      g = out_image.at<cv::Vec3b>(i,j)[1] / 255.0; 
      r = out_image.at<cv::Vec3b>(i,j)[2] / 255.0;

      out_image.at<cv::Vec3b>(i,j)[0] = get_h(r, g, b);
      out_image.at<cv::Vec3b>(i,j)[1] = (1 - (3 / (r + g + b)) * std::min(std::min(r, g), b)) * 255;
      out_image.at<cv::Vec3b>(i,j)[2] = std::max(r, std::max(g, b)) * 255;
    }
  }
}

void RGB2HSI(cv::Mat out_image)
{
  float r, g, b;

  for(int i = 0; i < out_image.rows; i++) {
    for(int j = 0; j < out_image.cols; j++ ) { 
      b = out_image.at<cv::Vec3b>(i,j)[0] / 255.0; 
      g = out_image.at<cv::Vec3b>(i,j)[1] / 255.0; 
      r = out_image.at<cv::Vec3b>(i,j)[2] / 255.0;
    
      out_image.at<cv::Vec3b>(i,j)[0] = get_h(r, g, b);
      out_image.at<cv::Vec3b>(i,j)[1] = (1 - (3 / (r + g + b)) * std::min(std::min(r, g), b)) * 255;
      out_image.at<cv::Vec3b>(i,j)[2] = (r + g + b) / 3 * 255;
    }
  }
}

void RGB2HSI_openCV(cv::Mat out_image, cv::Mat in_image)
{
  cv::cvtColor(out_image, out_image, cv::COLOR_BGR2HSV);

  for(int i = 0; i < out_image.rows; i++) {
    for(int j = 0; j < out_image.cols; j++ ) { 
      out_image.at<cv::Vec3b>(i, j)[2] = (in_image.at<cv::Vec3b>(i,j)[0] + in_image.at<cv::Vec3b>(i,j)[1] + in_image.at<cv::Vec3b>(i,j)[2]) / 3;
    }
  }
}

cv::Mat image_processing(const cv::Mat in_image, int key) 
{
  // Create output image
  cv::Mat out_image;;

  // Copy the original image
  in_image.copyTo(out_image);

  switch(key) {
    case OP2:
      RGB2CMY(out_image);
      break;
    case OP3:
      RGB2HSI(out_image); 
      break;
    case OP4:
      RGB2HSV(out_image);
      break;
    case OP5:
      cv::cvtColor(out_image, out_image, cv::COLOR_BGR2HSV);
      break;
    case OP6:
      RGB2HSI_openCV(out_image, in_image);
  }

  // Write text in an image
  cv::putText(out_image, "1: RGB, 2: CMY, 3: HSI, 4: HSV, 5: HSV OpenCV, 6: HSI OpenCV", 
              cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  // Show image 
  cv::imshow("P1", out_image);

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}