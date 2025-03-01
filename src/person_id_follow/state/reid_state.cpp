#include "person_id_follow/state/reid_state.hpp"

#include "person_id_follow/state/tracking_state.hpp"

#include <geometry_msgs/Point.h>

namespace person_id_follow {

State* ReidState::update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) {
    std::cout << "---------------------" << std::endl;
    std::cout << "\033[1;33mReidState \033[0mstart!" << std::endl;
    std::cout << "---------------------" << std::endl;

    // 追加修正by mukogawa
    long track_number = -1;

    for(const auto& track: tracks) {
        ROS_INFO_STREAM("reid state predict start!!");
        // 修正by mukogawa
        track_number = track.first;
        boost::optional<double> pred = context.predict(track_number, track.second);
        if(!pred) {
            ROS_INFO_STREAM("--- !pred ---");
            continue;
        }
        // if(pred < nh.param<double>("reid_confidence_thresh", 0.2)) {
        //     ROS_INFO_STREAM("--- pred < reid_confidence_thresh ---");
        //     continue;
        // }
        // FIXME : Adjustment of reid_confidence_thresh
        if(pred < nh.param<double>("reid_confidence_thresh", 0.2)) {
            ROS_INFO_STREAM("--- pred < reid_confidence_thresh ---");
            continue;
        }

        // FIXME : Adjustment of reid_positive_confidence_thresh
        if(pred > nh.param<double>("reid_positive_confidence_thresh", 0.8)) {
            ROS_INFO_STREAM("--- pred > 0.8, tracking state ---");
            return new TrackingState(track.first);
        }

        auto found = positive_count.find(track.first);
        if(found == positive_count.end()) {
            positive_count[track.first] = 0;
        }
        positive_count[track.first] ++;

        std::cout << "positive_count[track.first] : " << positive_count[track.first] << std::endl;
        if(positive_count[track.first] >= nh.param<int>("reid_positive_count", 5)) {
            return new TrackingState(track.first);
        }
    }

    return this;
}

}
