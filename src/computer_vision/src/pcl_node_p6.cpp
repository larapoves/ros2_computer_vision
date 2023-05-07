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

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

enum {
  MIN_POINTS = 100
};

cv::Mat T = cv::Mat::zeros(3, 4, CV_32F); // Extrinsic parameters

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);
pcl::PointCloud<pcl::PointXYZRGB> color_filter(const pcl::PointCloud<pcl::PointXYZRGB> cloud);
pcl::PointCloud<pcl::PointXYZRGB> draw_square(pcl::PointCloud<pcl::PointXYZRGB> cloud, const cv::Scalar color, const float x, const float y, const float z);
pcl::PointCloud<pcl::PointXYZRGB> detect_balls(const pcl::PointCloud<pcl::PointXYZRGB> cloud_balls, pcl::PointCloud<pcl::PointXYZRGB> cloud);
pcl::PointCloud<pcl::PointXYZRGB> distance(pcl::PointCloud<pcl::PointXYZRGB> cloud);

class PCLSubscriber : public rclcpp::Node
{
  public:
    PCLSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS(rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
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

      // PCL Processing
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

pcl::PointCloud<pcl::PointXYZRGB> color_filter(const pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pink (new pcl::PointCloud<pcl::PointXYZRGB>);

  for (const auto& point: cloud) {
    if(point.r >= 35 && point.b >= 35 && point.g <= 38) {
      pink->push_back(point);
    }
  }

  // Remove outliers values if there are any ball
  if(pink->size() > MIN_POINTS) {
    sor.setInputCloud(pink);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(cloud_filtered);
  }
  
  return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_square(pcl::PointCloud<pcl::PointXYZRGB> cloud, const cv::Scalar color, const float x, const float y, const float z)
{
  float dist = 0.01, side = 0.1;
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

pcl::PointCloud<pcl::PointXYZRGB> detect_balls(const pcl::PointCloud<pcl::PointXYZRGB> cloud_balls, pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_p (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  pcl::SACSegmentation<pcl::PointXYZRGB> seg; // Segmentation object
  pcl::ExtractIndices<pcl::PointXYZRGB> extract; // Filtering object

  *cloud_ptr = cloud_balls;
  if(cloud_ptr->size() <= MIN_POINTS) {
    std::cerr << "There isn't any pink ball." << std::endl;
    return cloud;
  }

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_SPHERE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(0.01);

  while(cloud_ptr->size() > MIN_POINTS)
  {
    // Segment the largest sphere component from the remaining cloud
    seg.setInputCloud(cloud_ptr);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.size () == 0) {
      std::cerr << "Could not estimate a sphere model for the given dataset." << std::endl;
      break;
    }

    // Draw a square in the center of the sphere
    cloud = draw_square(cloud, cv::Scalar(255, 0, 0), coefficients->values[0], coefficients->values[1], coefficients->values[2]);

    // Extract the inliers
    extract.setInputCloud(cloud_ptr);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_p);

    // Create the filtering object
    extract.setNegative(true);
    extract.filter(*cloud_f);
    cloud_ptr.swap(cloud_f);
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> distance(pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
  cv::Mat gazebo = (cv::Mat_<float>(4, 1) << 0.0, 1.0, 0.0, 1.0);
  cv::Mat camera = cv::Mat::zeros(3, 1, CV_32F);

  for(int s = 0; s < 2; s++) {
    for(int i = 3; i <= 8; i++) {
      // Get the coordinates
      gazebo.at<float>(0, 0) = i; 
      camera = T * gazebo;

      // Draw the square
      cv::Scalar color = cv::Scalar(0, (i - 3) * 51, 255 - (i - 3) * 51);
      cloud = draw_square(cloud, color, camera.at<float>(0, 0), camera.at<float>(1, 0), camera.at<float>(2, 0));
    }
    gazebo.at<float>(1, 0) = -1.0;
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud, balls_pointcloud;

  // Processing
  out_pointcloud = in_pointcloud;

  // Color filter pink
  balls_pointcloud = color_filter(out_pointcloud);
  out_pointcloud = detect_balls(balls_pointcloud, out_pointcloud);
  out_pointcloud = distance(out_pointcloud);

  return out_pointcloud;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PCLSubscriber>());
  rclcpp::shutdown();
  return 0;
}