#!/bin/bash

echo "╔══╣ Install: People Tracking (STARTING) ╠══╗"


# Get current directory
DIR=`pwd`

# Download dependencies
sudo apt-get install -y \
    ros-$ROS_DISTRO-usb-cam \
    ros-$ROS_DISTRO-image-view \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-message-filters \
    ros-$ROS_DISTRO-eigen-conversions

# Clone required repositories
cd ../
git clone https://github.com/TeamSOBITS/monocular_people_tracking.git
cd monocular_people_tracking/
bash install.sh


# Return to the original directory
cd $DIR


echo "╚══╣ Install: People Tracking (FINISHED) ╠══╝"
