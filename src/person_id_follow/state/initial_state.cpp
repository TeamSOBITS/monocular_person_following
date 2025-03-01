#include <math.h>
#include "person_id_follow/state/initial_state.hpp"
#include "person_id_follow/state/initial_training_state.hpp"

#include <geometry_msgs/Point.h>

namespace person_id_follow {

State* InitialState::update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) {
    // Select one target with select_target
    std::cout << "---------------------" << std::endl;
    std::cout << "\033[1;33mInitialState \033[0mstart!" << std::endl;
    std::cout << "---------------------" << std::endl;
    long target_id = select_target(nh, tracks);

    if(target_id < 0) {
        return this;
    }

    return new InitialTrainingState(target_id);
}

long InitialState::select_target(ros::NodeHandle& nh, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks) {
    long target_id = -1;
    double distance = 0.0;
    // ROS_INFO_STREAM("initial state select_target start!");
    for(const auto& track: tracks) {
        float pose_x = track.second->pose_x;
        float pose_y = track.second->pose_y;
        // Skip if x distance is greater than max_dist(4.0m)
        if(pose_x > nh.param<double>("imprinting_max_dist", 4.0)) {
            continue;
        }
        // Update target_id and distance
        // target_id is the nearest person from the robot
        if(target_id == -1 || distance > sqrt(pose_x * pose_x + pose_y * pose_y)) {
            target_id = track.first;
            distance = sqrt(pose_x * pose_x + pose_y * pose_y);
        }
    }
    std::cout << "initial_state target_id : " << target_id << std::endl;

    return target_id;
}

}
