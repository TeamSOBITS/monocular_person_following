cmake_minimum_required(VERSION 3.10.2)
project(monocular_person_following)

add_compile_options(-std=c++14)

find_package(OpenCV 4.9.0 REQUIRED PATHS "/opt/opencv/opencv_4.9.0/")

find_package(catkin REQUIRED
  rospy
  roscpp
  message_generation
  cv_bridge
  tf
  message_filters
  image_transport
  eigen_conversions
  usb_cam
  image_view
  rosbag
  sensor_msgs
  geometry_msgs
  ccf_person_identification
  monocular_people_tracking
  lightweight_human_pose_estimation
)

########################
## service generation ##
########################
add_service_files(FILES
  Imprint.srv
)

########################
## message generation ##
########################
add_message_files(FILES
  BoundingBox2D.msg
  FaceDetection.msg
  FaceDetectionArray.msg
  Target.msg
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

###################################
## catkin specific configuration ##
###################################
catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES monocular_person_following
    #  CATKIN_DEPENDS other_catkin_pkg
    #  DEPENDS system_lib
    )

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(monocular_person_following_node
  src/monocular_person_following/state/initial_state.cpp
  src/monocular_person_following/state/initial_training_state.cpp
  src/monocular_person_following/state/tracking_state.cpp
  src/monocular_person_following/state/reid_state.cpp
  src/monocular_person_following/context.cpp
  src/monocular_person_following_node.cpp
)
add_dependencies(monocular_person_following_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(monocular_person_following_node
  ${catkin_LIBRARIES}
)
