cmake_minimum_required(VERSION 3.0.2)
project(person_id_follow_nodelet)

add_compile_options(-std=c++14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
# include(flann/cmake)

# find_package：フォルダを検索する
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Flann REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  cv_bridge
  message_filters
  image_transport
  message_generation
  std_msgs
  roslib
  eigen_conversions
  ccf_person_identification
  pluginlib
  nodelet
  geometry_msgs
)

########################
## service generation ##
########################
# add_service_files(FILES
#   Imprint.srv
# )

########################
## message generation ##
########################
add_message_files(FILES
  SOBITTarget.msg
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs geometry_msgs)

include(dlib/CMakeLists.txt)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES person_id_follow_nodelet
  CATKIN_DEPENDS message_runtime
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)


include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${FLANN_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(person_id_follow_nodelet
  src/person_id_follow/state/init_state.cpp
  src/person_id_follow/state/initial_state.cpp
  src/person_id_follow/state/initial_training_state.cpp
  src/person_id_follow/state/tracking_state.cpp
  src/person_id_follow/state/reid_state.cpp
  src/person_id_follow/sobit_context_init.cpp
  src/person_id_follow_nodelet.cpp
)

###########
## Build ##
###########

target_link_libraries(person_id_follow_nodelet
  lz4
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${FLANN_LIBRARIES}
  dlib::dlib
)