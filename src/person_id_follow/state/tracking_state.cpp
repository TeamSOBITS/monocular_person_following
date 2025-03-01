#include "person_id_follow/state/tracking_state.hpp"

#include "person_id_follow/state/reid_state.hpp"

#include <geometry_msgs/Point.h>

namespace person_id_follow {

//たいちさんのコードから修正したby mukogawa(2024/8/16)
State* TrackingState::update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) {
    std::cout << "---------------------" << std::endl;
    std::cout << "\033[1;33mTrackingState \033[0mstart!" << std::endl;
    std::cout << "---------------------" << std::endl;

    // 初期状態
    double max_pred = std::numeric_limits<double>::lowest();
    long best_target_id = -1;

    // 距離ベースで最も近い人物IDを取得
    double min_distance = std::numeric_limits<double>::max();
    long closest_target_id = -1;
    bool DistanceBase_targetID = false;

    // 対象者のみの状態から、非対象者も加わって初期分類が行われる際、上手く初期分類できず非対象者を誤識別してしまう課題があるため、最初の何フレームかは距離ベースで対象者を特定して正しく対象者を分類できるようにする補助的な処理
    if(tracks.size() >= 2 && target_InitialClassifier_count < 5) {
        for(const auto& track: tracks) {
            long track_id = track.first;

            float pose_x = track.second->pose_x;
            float pose_y = track.second->pose_y;

            float distance = std::hypotf(pose_x - target_position_keep.x, pose_y - target_position_keep.y);

            // 各trackのdistanceとtrack_idを可視化
            std::cout << "Track ID: " << track_id 
                    << ", Distance: " << distance 
                    << ", Pose (x, y): (" << pose_x << ", " << pose_y << ")"
                    << std::endl;

            if(distance < min_distance) {
                min_distance = distance;
                closest_target_id = track_id;
            }
        }

        DistanceBase_targetID = true;
        target_InitialClassifier_count++;

    }
    
    for(const auto& track : tracks) {
        if(DistanceBase_targetID == true) {
            // 各トラックごとのpredを計算
            long track_id = track.first;

            if(track_id == closest_target_id) {
                
                // closest_target_idに基づいてmax_pred,best_target_idを更新
                boost::optional<double> pred = context.predict(track_id, track.second);
                max_pred = *pred;
                best_target_id = track_id;
                break;

            }
        }
        else{
            // 各トラックごとのpredを計算
            long track_id = track.first;

            boost::optional<double> pred = context.predict(track_id, track.second);

            // track_id と pred の値を表示
            std::cout << "Track ID: " << track_id 
                    << ", Predicted Value: " << (pred ? std::to_string(*pred) : "None") 
                    << std::endl;
            
            // predが存在し、現在の最大値よりも大きい場合は更新
            if(pred && *pred > max_pred) {
                max_pred = *pred;
                best_target_id = track_id;
            }
        }
    }

    // best_target_idが見つかった場合、そのtarget_idに更新
    if(best_target_id != -1) {
        target_id = best_target_id;
    }
    

    auto found = tracks.find(target_id);

    if(found == tracks.end()) {
        ROS_INFO_STREAM("target lost!!");
        return new ReidState();
    }

    // FIXME：Adjustment of "id_switch_detection_thresh"
    if(max_pred < nh.param<double>("id_switch_detection_thresh", -0.1)) {
        ROS_INFO_STREAM("ID switch detected!!");
        return new ReidState();
    }

    // FIXME:Adjustment of "min_target_cofidence"
    if(max_pred < nh.param<double>("min_target_cofidence", 0.2)) {
        ROS_INFO_STREAM("do not update for pred < min_target_cofidence");
        return this;
    }

    for(const auto& track: tracks) {
        double label = track.first == target_id ? 1.0 : -1.0;
        std::cout << "tracking_state label : " << label << std::endl;
        context.update_classifier(label, track.first, track.second);
    }


    return this;
}


}
