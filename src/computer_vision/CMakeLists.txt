cmake_minimum_required(VERSION 3.8)
project(computer_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#######

# OpenCV node
add_executable(cv_node src/cv_node.cpp)
target_include_directories(cv_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
)

target_link_libraries(cv_node ${PCL_LIBRARIES})

install(TARGETS cv_node
  DESTINATION lib/${PROJECT_NAME})

# OpenCV node color filter
add_executable(cv_node_color_filter src/cv_node_color_filter.cpp)
target_include_directories(cv_node_color_filter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node_color_filter PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node_color_filter
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
)

target_link_libraries(cv_node_color_filter ${PCL_LIBRARIES})

install(TARGETS cv_node_color_filter
  DESTINATION lib/${PROJECT_NAME})

# OpenCV node P1
add_executable(cv_node_p1 src/cv_node_p1.cpp)
target_include_directories(cv_node_p1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node_p1 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node_p1
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
)

target_link_libraries(cv_node_p1 ${PCL_LIBRARIES})

install(TARGETS cv_node_p1
  DESTINATION lib/${PROJECT_NAME})

# OpenCV node P2
add_executable(cv_node_p2 src/cv_node_p2.cpp)
target_include_directories(cv_node_p2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node_p2 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node_p2
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
)

target_link_libraries(cv_node_p2 ${PCL_LIBRARIES})

install(TARGETS cv_node_p2
  DESTINATION lib/${PROJECT_NAME})

# OpenCV node P3
add_executable(cv_node_p3 src/cv_node_p3.cpp)
target_include_directories(cv_node_p3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node_p3 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node_p3
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
)

target_link_libraries(cv_node_p3 ${PCL_LIBRARIES})

install(TARGETS cv_node_p3
  DESTINATION lib/${PROJECT_NAME})

# OpenCV node P4
add_executable(cv_node_p4 src/cv_node_p4.cpp)
target_include_directories(cv_node_p4 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node_p4 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node_p4
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
)

target_link_libraries(cv_node_p4 ${PCL_LIBRARIES})

install(TARGETS cv_node_p4
  DESTINATION lib/${PROJECT_NAME})

# OpenCV node P5
add_executable(cv_node_p5 src/cv_node_p5.cpp)
target_include_directories(cv_node_p5 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_node_p5 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_node_p5
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
  "image_geometry"
  "tf2_ros"
)

target_link_libraries(cv_node_p5 ${PCL_LIBRARIES} ${image_geometry_LIBRARIES} ${tf2_ros_LIBRARIES})

install(TARGETS cv_node_p5
  DESTINATION lib/${PROJECT_NAME})

# PCL node
add_executable(pcl_node src/pcl_node.cpp)
target_include_directories(pcl_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pcl_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  pcl_node
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
  "pcl_conversions"
)

target_link_libraries(pcl_node ${PCL_LIBRARIES})

install(TARGETS pcl_node
  DESTINATION lib/${PROJECT_NAME})

# PCL node P6
add_executable(pcl_node_p6 src/pcl_node_p6.cpp)
target_include_directories(pcl_node_p6 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pcl_node_p6 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  pcl_node_p6
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
  "pcl_conversions"
  "tf2_ros"
)

target_link_libraries(pcl_node_p6 ${PCL_LIBRARIES} ${tf2_ros_LIBRARIES})

install(TARGETS pcl_node_p6
  DESTINATION lib/${PROJECT_NAME})

# OpenCV and PCL node
add_executable(cv_pcl_node src/cv_pcl_node.cpp)
 target_include_directories(cv_pcl_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_pcl_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  cv_pcl_node
  "rclcpp"
  "image_transport"
  "cv_bridge"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "PCL"
  "pcl_conversions"
  "image_geometry"
  "tf2_ros"
)
  
target_link_libraries(cv_pcl_node ${PCL_LIBRARIES} ${image_geometry_LIBRARIES} ${tf2_ros_LIBRARIES})
  
install(TARGETS cv_pcl_node
  DESTINATION lib/${PROJECT_NAME})

  
install(DIRECTORY launch config params maps
        DESTINATION share/${PROJECT_NAME})

#######

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
