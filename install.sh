#!/bin/bash

echo "╔══╣ Install: People Tracking (STARTING) ╠══╗"


# Get current directory
DIR=`pwd`

# Download dependencies
sudo apt-get install -y \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-tf \
    ros-$ROS_DISTRO-message-filters \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-eigen-conversions \
    ros-$ROS_DISTRO-usb-cam \
    ros-$ROS_DISTRO-image-view \
    ros-$ROS_DISTRO-rosbag \
    ros-$ROS_DISTRO-sensor-msgs \
    ros-$ROS_DISTRO-geometry-msgs

# Clone required repositories
cd ../
git clone --recurse-submodules https://github.com/TeamSOBITS/monocular_people_tracking.git
cd monocular_people_tracking/
bash install.sh

# Install python dependencies
sudo apt update
sudo apt install -y \
    python3-pip

python3 -m pip install --upgrade pip
python3 -m pip install numpy

# Return to the original directory
cd $DIR


echo "╚══╣ Install: People Tracking (FINISHED) ╠══╝"
