#include "person_id_follow/state/initial_training_state.hpp"

#include "person_id_follow/state/initial_state.hpp"
#include "person_id_follow/state/tracking_state.hpp"

#include <geometry_msgs/Point.h>

namespace person_id_follow {

State* InitialTrainingState::update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) {
    std::cout << "---------------------" << std::endl;
    std::cout << "\033[1;33mInitalTrainingState \033[0mstart!" << std::endl;
    std::cout << "---------------------" << std::endl;
    
    // auto found = tracks.find(target_id);
    // if(found == tracks.end()) {
    //     lost_count ++;
    //     if (lost_count > 100){
    //         ROS_INFO_STREAM("lost target during the initial training!!");
    //         return new InitialState();
    //     }
    //     return this;
    // }

    long target_id = -1;
    double distance = 0.0;
    // ROS_INFO_STREAM("initial state select_target start!");
    for(const auto& track: tracks) {
        float pose_x = track.second->pose_x;
        float pose_y = track.second->pose_y;
        // std::cout << "pose_x : " << pose_x << std::endl;
        // std::cout << "pose_y : " << pose_y << std::endl;
        // Skip if pose_x or pose_y is zero（ssdでobj_poseの値が距離が遠くて入らなかった人の場合）
        if (pose_x == 0.0 && pose_y == 0.0) {
            continue;
        }
        // Skip if x distance is greater than max_dist(4.0m)
        if(pose_x > nh.param<double>("imprinting_max_dist", 4.0)) {
            continue;
        }
        // Update target_id and distance
        // target_id is the person closest to the robot
        if(target_id == -1 || distance > sqrt(pose_x * pose_x + pose_y * pose_y)) {
            target_id = track.first;
            distance = sqrt(pose_x * pose_x + pose_y * pose_y);
        }
    }
    if(target_id == -1) {
        return this;
    }


    // 追加修正by mukogawa
    long track_number = -1;

    for(const auto& track: tracks) {
        // If track.first matches target_id : label = 1.0
        // If track.first mismatches target_id: label = -1.0
        // If track is the id of the nearest person, set label to 1.0, otherwise set to -1.0
        double label = track.first == target_id ? 1.0 : -1.0;
        // Related function: predict in context.cpp => predict in person_classifier.hpp
        // 修正by mukogawa
        track_number = track.first;
        boost::optional<double> pred = context.predict(track_number, track.second);
        if (!pred){
            ROS_INFO_STREAM("initial_trainig_state predict failed for !pred");
        }
        // Reference: update_classifier in context.cpp
        // For label of update_classifier, put 1.0 for target and -1.0 for others.
        bool update_classifier_flag = context.update_classifier(label, track_number, track.second);
        // if(!update_classifier_flag){
        //     ROS_INFO_STREAM("update_classifier_flag failed");
        // }

        if(label > 0.0) {
            num_pos_samples ++;
        }
    }

    // if(num_pos_samples >= nh.param<int>("initial_training_num_samples", 300)) {
    //     return new TrackingState(target_id);
    // }

    if(num_pos_samples >= nh.param<int>("initial_training_num_samples", 50)) {
        return new TrackingState(target_id);
    }

    return this;
}

}
