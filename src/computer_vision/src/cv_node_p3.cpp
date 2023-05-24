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

// trasformaciones elementales
  // esclado [x y 1]t = [sx 0 0; 0 sy 0; 0 0 1] * [i j 1]t
  // translación [x y 1]t = [1 0 0; 0 1 0; tx ty 1] * [i j 1]t
  // rotacion [x y 1]t = [cos -sen 0; sen cos 0; 0 0 1] * [i j 1]t
  // inclinación [x y 1]t = [1 teta 0; teta 1 0; 0 0 1] * [i j 1]t

// interpolación vecinos -> hacer más grande poniendo el color del pixel original a los de al lado -> efecto cuadriculado

// suavizado de vecinos como una convalución
// filtro de media -> 1/9 [matriz de 1]

// El histograma es la representación gráfica de las frecuencias relativas con las que aparecen los distintos colores en una determinada imagen

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

enum{
  RADIUS_FILTER = 50
};

int key = '1';
int c_min = 0;
int c_max = 30;
bool show = false;

cv::Mat image_processing(const cv::Mat in_img);

cv::Mat computeDFT(const cv::Mat &image);
cv::Mat fftShift(const cv::Mat &magI);
cv::Mat low_pass_filter(const cv::Mat gray_img);

cv::Mat stretch_shrink(const cv::Mat img, const bool shrink);
void show_histograms(const std::vector<cv::Mat> imgs);
cv::Mat enhanced(const cv::Mat image);

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

cv::Mat low_pass_filter(const cv::Mat gray_img)
{
  // Transform of Fourier
  cv::Mat complex_img = computeDFT(gray_img);

  // Shift quadrants
  cv::Mat shift_complex = fftShift(complex_img);

  // Draw a circle of radius 50
  cv::Mat filter = cv::Mat::zeros(cv::Size(shift_complex.cols, shift_complex.rows), CV_32F);
  cv::circle(filter, cv::Point(filter.cols / 2, filter.rows / 2), RADIUS_FILTER, cv::Scalar(1.0), -1);

  // Filter of two channels
  cv::Mat filter_2c = cv::Mat::zeros(cv::Size(shift_complex.cols, shift_complex.rows), CV_32FC2);
  std::vector<cv::Mat> channels;
  cv::split(filter_2c, channels);

  // Assign the low pass filter to both channels
  channels[0] = filter;
  channels[1] = filter;

  // Merge
  cv::merge(channels, filter_2c);

  // Multiply Fourier and filter
  cv::mulSpectrums(shift_complex, filter_2c, shift_complex, 0);

  // Shift quadrants
  cv::Mat final_shift = fftShift(shift_complex);

  // Calculate the idft
  cv::Mat inverse;
  cv::idft(final_shift, inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(inverse, inverse, 0, 255, cv::NORM_MINMAX, CV_MAT_DEPTH(CV_8UC1));

  return inverse;
}

void show_histograms(const std::vector<cv::Mat> imgs)
{
  // Histograms data
  std::vector<cv::Mat> histograms(5);
  std::vector<std::string> text(4);
  std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0),
                                    cv::Scalar(0, 255, 255), cv::Scalar(0, 255, 0)};

  // Establish the number of bins
  int histSize = 256;

  // Set the ranges
  float range[] = {0, 256};
  const float* histRange = {range};
  bool uniform = true, accumulate = false;

  // Draw the histogram
  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound((double) hist_w / histSize);
  cv::Mat hist_img(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

  // Text
  text[0] = "Shrink [" + std::to_string(c_min) + ", " + std::to_string(c_max) + "]: ";
  text[1] = "Subtract: ";
  text[2] = "Stretch: ";
  text[3] = "Eqhist: ";
  
  for(int i = 0, h = 20; i < 5; i++) {
    // Compute and normalize histogram
    cv::calcHist(&imgs[i], 1, 0, cv::Mat(), histograms[i], 1, &histSize, &histRange, uniform, accumulate);
    cv::normalize(histograms[i], histograms[i], 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Draw the intensity line for the histogram
    for(int j = 1; j < histSize; j++) {
      cv::line(hist_img, cv::Point(bin_w * (j - 1), hist_h - cvRound(histograms[i].at<float>(j - 1))),
              cv::Point(bin_w*(j), hist_h - cvRound(histograms[i].at<float>(j))), colors[i], 2, 8, 0); 
    }

    // Put text
    if(i != 0) {
      text[i - 1] += std::to_string(cv::compareHist(histograms[0], histograms[i], CV_COMP_CORREL)); // CV_COMP_CHISQR, CV_COMP_INTERSECT, CV_COMP_BHATTACHARYYA
      cv::putText(hist_img, text[i - 1], cv::Point(10, h), cv::FONT_HERSHEY_SIMPLEX, 0.45, colors[i]); 
      h += 20;
    }
  }

  cv::imshow("Histogram", hist_img);
  show = true;
}

// expandir -> distribuir las frecuencias a lo largo de todo el histogrma -> 0 255
// cmax y cmin son los valores deseados de la compresión
// rmxa y rmin máximo y mínimo nivel de gris de la imagen
cv::Mat stretch_shrink(const cv::Mat img, const bool shrink)
{
  // Find maximum and minimum values of the filtered image
  double r_min, r_max;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(img, &r_min, &r_max, &min_loc, &max_loc);

  uint sub, min;
  if(shrink) {
    sub = c_max - c_min;
    min = c_min;
  } else {
    sub = 255;
    min = 0;
  }

  // Shrink or Stretch
  cv::Mat out_img = img.clone();
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      out_img.at<uchar>(i, j) = (uchar) ((sub) / (r_max - r_min) * (img.at<uchar>(i, j) - r_min) + min);
    }
  }

  return out_img;
}

cv::Mat enhanced(const cv::Mat in_img)
{
  std::vector<cv::Mat> imgs(5);

  // Convert to gray scale
  cv::cvtColor(in_img, imgs[0], cv::COLOR_BGR2GRAY); 
  
  // Low pass filter
  cv::Mat filter_img = low_pass_filter(imgs[0]);

  // Shrink
  imgs[1] = stretch_shrink(filter_img, true);

  // Subtract 
  cv::subtract(imgs[0], imgs[1], imgs[2]);

  // Stretch
  imgs[3] = stretch_shrink(imgs[2], false);

  // Equalization
  cv::equalizeHist(imgs[3], imgs[4]);

  // Display the histograms
  show_histograms(imgs);

  return imgs[4];
}

cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image;
  
  // Processing
  out_image = in_image;

  // Read key
  int key_read = cv::waitKey(1); // waiting time for a key pressed
  if(key_read >= '1' && key_read <= '3') {
    key = key_read;
  }
  
  // Control min-max values
  if(key_read == 'z' && c_min > 0) {
    c_min--;
  } else if (key_read == 'x' && c_min < c_max - 1) {
    c_min++;
  } else if(key_read == 'v' && c_max < 255){
    c_max++;
  } else if(key_read == 'c' && c_max > c_min + 1) {
    c_max--;
  }

  if(show && key != '3') {
    cv::destroyWindow("Histogram");
    show = false;
  }
  
  switch(key) {
    case '2':
      // Convert to gray scale
      cv::cvtColor(out_image, out_image, cv::COLOR_BGR2GRAY);
      break;
    case '3':
      out_image = enhanced(out_image);
  }

  // Create a rectangle
  cv::rectangle(out_image, cv::Point(0, 0), cv::Point(out_image.cols, 50), cv::Scalar(255, 255, 255), -1);

  // Put text
  std::string val = "shrink [min: " + std::to_string(c_min) + ", max: " + std::to_string(c_max) + "]";
  cv::putText(out_image, val, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255));
  cv::putText(out_image, "1: Original, 2: Gray, 3: Enhanced | shrink [z, x]: -+ min, shrink [c, v]: -+ max", 
              cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255));

  // Show image in a different window
  cv::imshow("out_image",out_image);

  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}