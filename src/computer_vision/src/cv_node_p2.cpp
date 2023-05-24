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
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// filtro de paso alto -> bordes
// filtro de paso bajo -> suaviza -> fondo negro, círculo blanco

// convlolución -> filter2D(imagen, imagen_convolucionada, imagen.depth(), kernel);
  // Kernel es la matriz h, matriz de convolución
    // Masks
      /*
      Mat M1 = (Mat_<char>(3,3) <<
      1, 1, 1,
      1, 1, 1,
      1, 1, 1);
      */
  // imagen es la imagen original
  // imagen_conlolución es donde guardamos el resultado

// transformaciones de vecindad
  // el pixel tiene el valor promedio de los 8 pixeles que tiene alrededor
  // se pueden aplicar máscaras para decidir el peso de cada uno
  // se hace como la convolución

enum {
  MIN_FILTER = 50,
  MAX_FILTER = 100,
};

int key = '1';
bool opd = false;
bool show = false;
int thickness = MIN_FILTER;

cv::Mat image_processing(const cv::Mat in_image);

cv::Mat computeDFT(const cv::Mat &image);
cv::Mat fftShift(const cv::Mat &magI);
cv::Mat spectrum(const cv::Mat &complexI);

cv::Mat cross_filter(const bool keep, const cv::Size size);
cv::Mat frecuenciesHV(const cv::Mat image, const bool keep, const bool display);

cv::Mat threshold(const cv::Mat img, const float value);
cv::Mat AND(const cv::Mat image);

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
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw = cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; 
      img_bridge.toImageMsg(out_image); 

      // Publish the data
      publisher_ -> publish(out_image);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

cv::Mat computeDFT(const cv::Mat &image) 
{
  // Expand the image to an optimal size. 
  cv::Mat padded;                      
  int m = cv::getOptimalDFTSize(image.rows);
  int n = cv::getOptimalDFTSize(image.cols); 
  cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // Make place for both the complex and the real values
  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);   

  // Make the Discrete Fourier Transform
  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT); 

  return complexI; 
}

cv::Mat fftShift(const cv::Mat &magI) 
{
  cv::Mat magI_copy = magI.clone();

  // Crop the spectrum, if it has an odd number of rows or columns
  magI_copy = magI_copy(cv::Rect(0, 0, magI_copy.cols & -2, magI_copy.rows & -2));
  
  // Rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI_copy.cols/2;
  int cy = magI_copy.rows/2;

  cv::Mat q0(magI_copy, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(magI_copy, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(magI_copy, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(magI_copy, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

  cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return magI_copy;
}

cv::Mat spectrum(const cv::Mat &complexI) // sirve para visualizar la transformada de fourier
{
  cv::Mat complexImg = complexI.clone();

  // Shift quadrants
  cv::Mat shift_complex = fftShift(complexImg);

  cv::Mat planes_spectrum[2];
  split(shift_complex, planes_spectrum);      
  magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);
  cv::Mat spectrum = planes_spectrum[0];

  // Switch to a logarithmic scale
  spectrum += cv::Scalar::all(1);
  log(spectrum, spectrum);

  // Normalize
  cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX); 

  return spectrum;
}

cv::Mat cross_filter(const bool keep, const cv::Size size)
{
  cv::Mat filter;

  if(keep) {
    filter = cv::Mat::zeros(size, CV_32F);
    cv::line(filter, cv::Point(filter.cols / 2, 0), cv::Point(filter.cols / 2, filter.rows), cv::Scalar(1.0, 1.0, 1.0), thickness, cv::LINE_8, 0);
    cv::line(filter, cv::Point(0, filter.rows / 2), cv::Point(filter.cols, filter.rows / 2), cv::Scalar(1.0, 1.0, 1.0), thickness, cv::LINE_8, 0);
  } else {
    filter = cv::Mat::ones(size, CV_32F);
    cv::line(filter, cv::Point(filter.cols / 2, 0), cv::Point(filter.cols / 2, filter.rows), cv::Scalar(0.0, 0.0, 0.0), thickness,cv::LINE_8, 0);
    cv::line(filter, cv::Point(0, filter.rows / 2), cv::Point(filter.cols, filter.rows / 2), cv::Scalar(0.0, 0.0, 0.0), thickness, cv::LINE_8, 0);
  }

  return filter;
}

cv::Mat frecuenciesHV(const cv::Mat image, const bool keep, const bool display)
{
  // Transform of Fourier
  cv::Mat complex_img = computeDFT(image);

  // Shift quadrants
  cv::Mat shift_complex = fftShift(complex_img);

  // Filter of two channels
  cv::Mat filter_2c = cv::Mat::zeros(cv::Size(shift_complex.cols, shift_complex.rows), CV_32FC2);
  std::vector<cv::Mat> channels;
  cv::split(filter_2c, channels);

  // Assign the cross filter to both channels
  channels[0] = cross_filter(keep, cv::Size(filter_2c.cols, filter_2c.rows));
  channels[1] = cross_filter(keep, cv::Size(filter_2c.cols, filter_2c.rows));

  // Merge
  cv::merge(channels, filter_2c);

  // Multiply Fourier and filter
  cv::mulSpectrums(shift_complex, filter_2c, shift_complex, 0);

  // Shift quadrants
  cv::Mat final_shift = fftShift(shift_complex);

  // Option d
  if(display) {
    std::string name;
    if(keep) {
      name = "keep_filter";
    } else {
      name = "remove_filter";
    }

    cv::Mat spectrum_filtered = spectrum(final_shift);
    cv::imshow(name, spectrum_filtered);
  }

  // Calculate the idft
  cv::Mat inverse;
  cv::idft(final_shift, inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(inverse, inverse, 0, 1, cv::NORM_MINMAX);

  return inverse;
}

cv::Mat threshold(const cv::Mat img, const float value)
{
  cv::Mat out_img = img.clone();

  for(int i = 0; i < img.rows; i++) {
    for(int j = 0; j < img.cols; j++) {
      if(img.at<float>(i, j) > value) {
        out_img.at<float>(i, j) = 1.0;
      } else {
        out_img.at<float>(i, j) = 0.0;
      }
    }
  }

  return out_img;
}

cv::Mat AND(const cv::Mat image)
{
  cv::Mat op3 = frecuenciesHV(image, true, opd);
  cv::Mat op4 = frecuenciesHV(image, false, opd);

  cv::Mat threshols_op3 = threshold(op3, 0.4f);
  cv::Mat threshols_op4 = threshold(op4, 0.6f);

  if(opd){
    cv::imshow("keep_filter_bw", threshols_op3);
    cv::imshow("remove_filter_bw", threshols_op4);
    show = true;
  } else {
    if (show) {
      cv::destroyWindow("keep_filter");
      cv::destroyWindow("remove_filter");
      cv::destroyWindow("keep_filter_bw");
      cv::destroyWindow("remove_filter_bw");
      show = false;
    }
  }

  cv::Mat and_img;
  cv::bitwise_and(threshols_op4, threshols_op3, and_img); // transformción lógica, = para or y xor

  return and_img;
}

cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;

  // Processing image
  out_image = in_image;

  // Convert to gray scale
  cv::cvtColor(out_image, out_image, cv::COLOR_BGR2GRAY);

  // Read key
  int key_read = cv::waitKey(5); // waiting time for a key pressed
  if(key_read >= '1' && key_read <= '5') {
    key = key_read;
  }

  // Thickness
  if(key_read == 'z' && thickness > MIN_FILTER){
    thickness--;
  } else if(key_read == 'x' && thickness < MAX_FILTER) {
    thickness++;
  }

  // Option d
  if(key_read == 'd' && key == '5') {
    opd = !show;
  }

  switch(key) {
    case '2':
      out_image = computeDFT(out_image);
      out_image = spectrum(out_image);
      break;
    case '3':
      out_image = frecuenciesHV(out_image, true, false); // Keep filter
      break;
    case '4':
      out_image = frecuenciesHV(out_image, false, false);
      break;
    case '5':
      out_image = AND(out_image);
  }

  // Create a rectangle
  cv::rectangle(out_image, cv::Point(0, 0), cv::Point(out_image.cols, 50), cv::Scalar(255, 255, 255), -1);

  // Write text in an image
  std::string val = "[z, x]: -+ val: " + std::to_string(thickness);
  cv::putText(out_image, val, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  cv::putText(out_image, "1: GRAY, 2: Fourier, 3: Keep Filter, 4: Remove Filter, 5: AND", 
              cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

  // Show image 
  cv::imshow("P2", out_image);

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}