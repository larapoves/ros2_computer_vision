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
      cv::namedWindow("color filter");

      // Create the 3 Trackbars and add to a window
      cv::createTrackbar("low red", "color filter", nullptr, 255, 0);
      cv::createTrackbar("upper red", "color filter", nullptr, 255, 0);
      cv::createTrackbar("low blue", "color filter", nullptr, 255, 0);
      cv::createTrackbar("upper blue", "color filter", nullptr, 255, 0);
      cv::createTrackbar("low green", "color filter", nullptr, 255, 0);
      cv::createTrackbar("upper green", "color filter", nullptr, 255, 0);
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
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

      // Publish the data
      publisher_ -> publish(out_image);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
   
};

cv::Mat get_mask(const cv::Mat img, const cv::Scalar lower, const cv::Scalar upper)
{
  cv::Mat mask;

  cv::inRange(img, lower, upper, mask);

  return mask;
}

cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;

  // Get masks
  cv::Mat mask = get_mask(in_image, 
                          cv::Scalar(cv::getTrackbarPos("low blue", "color filter"), cv::getTrackbarPos("low green", "color filter"), cv::getTrackbarPos("low red", "color filter")), 
                          cv::Scalar(cv::getTrackbarPos("upper blue", "color filter"), cv::getTrackbarPos("upper green", "color filter"), cv::getTrackbarPos("upper red", "color filter")));

  in_image.copyTo(out_image,mask);

  // Show image in a different window
  cv::imshow("color filter",out_image);
  cv::waitKey(3);

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}