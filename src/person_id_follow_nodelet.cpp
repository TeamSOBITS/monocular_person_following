#include <iostream>
#include <ros/package.h>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include "ccf_person_identification/sobit_person_classifier.hpp"
#include "person_id_follow/sobit_tracklet.hpp"
#include "person_id_follow/sobit_context_init.hpp"
#include "person_id_follow/state/state.hpp"
#include "person_id_follow/state/init_state.hpp"
#include "person_id_follow/state/initial_state.hpp"
#include "person_id_follow/state/initial_training_state.hpp"
#include "person_id_follow_nodelet/SOBITTarget.h"
#include "sobits_msgs/ObjectPoseArray.h"
#include "sobits_msgs/BoundingBoxes.h"

#include <geometry_msgs/Point.h>

typedef message_filters::sync_policies::ApproximateTime<sobits_msgs::BoundingBoxes, sobits_msgs::ObjectPoseArray, sensor_msgs::Image> ImgSyncPolicy;

using namespace ccf_person_classifier;
using namespace person_id_follow;

namespace person_id_follow_nodelet{
    class CCFPersonId : public nodelet::Nodelet{
        private:
            ros::NodeHandle         nh_;
            ros::NodeHandle         pnh_;
            ros::Publisher          image_pub_;
            ros::Publisher          image_pub_1;
            cv_bridge::CvImagePtr   initial_cv_ptr;
            ros::Publisher          target_pub_;

            double previous_time_;
            geometry_msgs::Point target_position_keep;
            int target_InitialClassifier_count;

            image_transport::Publisher features_pub;

            std::mutex context_mutex;
            std::shared_ptr<State> state;
            std::unique_ptr<SOBITContextInit> context;
            
            std::unique_ptr<message_filters::Subscriber<sobits_msgs::BoundingBoxes>>  sub_bb_;
            std::unique_ptr<message_filters::Subscriber<sobits_msgs::ObjectPoseArray>>  sub_op_;
            std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>>               sub_sensor_img_;
            std::shared_ptr<message_filters::Synchronizer<ImgSyncPolicy>>                  sync_sensor_yolo_;
            void reset(long target_id);
        public:
            virtual void onInit();
            void InitialImageYOLOCb(const sobits_msgs::BoundingBoxesConstPtr &bb_msg,
                const sobits_msgs::ObjectPoseArrayConstPtr      &op_msg,
                const sensor_msgs::ImageConstPtr                     &img_msg);
            
    };
}

void person_id_follow_nodelet::CCFPersonId::onInit(){
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    image_pub_ = nh_.advertise<sensor_msgs::Image>("/person_id_follow_nodelet/person_id_img",1);
    image_pub_1 = nh_.advertise<sensor_msgs::Image>("/person_id_follow_nodelet/person_id_rect",1);
    image_transport::ImageTransport image_trans(nh_);
    features_pub = image_trans.advertise("/person_id_follow_nodelet/features", 1);
    target_pub_ = nh_.advertise<SOBITTarget>("/person_id_follow_nodelet/target", 1);

    previous_time_ = 0.0;
    target_position_keep.x = 0.0;
    target_position_keep.y = 0.0;
    target_InitialClassifier_count = 0; 

    sub_bb_.reset(new message_filters::Subscriber<sobits_msgs::BoundingBoxes>(nh_, pnh_.param<std::string>("yolo_bb_topic_name", "/yolov10_bbox_to_tf/object_rects"), 1 ) );
    sub_op_.reset(new message_filters::Subscriber<sobits_msgs::ObjectPoseArray>(nh_, pnh_.param<std::string>("yolo_op_topic_name", "/yolov10_bbox_to_tf/object_poses"), 1 ) );
    sub_sensor_img_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh_, pnh_.param<std::string>("sensor_img_topic_name", "/rgb/image_raw"), 1 ) );
    sync_sensor_yolo_.reset(new message_filters::Synchronizer<ImgSyncPolicy>( ImgSyncPolicy(10), *sub_bb_, *sub_op_, *sub_sensor_img_));
    sync_sensor_yolo_->registerCallback(boost::bind(&CCFPersonId::InitialImageYOLOCb, this, _1, _2, _3));
    state.reset(new InitialState());
    context.reset(new SOBITContextInit(nh_));
}

void person_id_follow_nodelet::CCFPersonId::InitialImageYOLOCb(const sobits_msgs::BoundingBoxesConstPtr &bb_msg,
                const sobits_msgs::ObjectPoseArrayConstPtr      &op_msg,
                const sensor_msgs::ImageConstPtr                     &img_msg){
    

    double dt = ( img_msg->header.stamp.toSec()  - previous_time_ );	//dt - expressed in seconds
    previous_time_ = img_msg->header.stamp.toSec();

    // 1ループあたりの処理時間を出力
    ROS_INFO("person_id_follow_nodelet_LOOP_TIME: dt = %.6f seconds!!!!!!!!!!!!!!!!!!", dt);

    std::unique_ptr<BodyClassifier> classifier(new BodyClassifier(nh_));
    cv::Mat img_raw;
    cv::Mat input_img;
    boost::optional<float> check_bb = bb_msg->bounding_boxes[0].probability;
    boost::optional<float> check_op = op_msg->object_poses[0].pose.position.x;
    bool pub_target_img;
    pub_target_img = pnh_.param<bool>( "pub_target_img", true );
    if (!check_bb){
        ROS_ERROR("bb_msg not found");
        return;
    }
    if (!check_op){
        ROS_ERROR("op_msg not found");
        return;
    }
    try{
        initial_cv_ptr = cv_bridge::toCvCopy( img_msg, sensor_msgs::image_encodings::BGR8 );
        img_raw = initial_cv_ptr->image.clone();
        input_img = initial_cv_ptr->image.clone();
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    } 
    if (input_img.empty() == true || img_raw.empty() == true){
        ROS_ERROR("input_img error");
        return;
    }

    // Check if bounding_boxes and object_poses have the same length
    if (bb_msg->bounding_boxes.size() != op_msg->object_poses.size()) {
        ROS_ERROR("Size mismatch: bounding_boxes size (%lu) != object_poses size (%lu)",
                bb_msg->bounding_boxes.size(), op_msg->object_poses.size());
        return;
    }
    
    // (1) xminを基準にbb_msg->bounding_boxesとop_msg->object_posesを対応させてソート
    std::vector<std::pair<sobits_msgs::BoundingBox, sobits_msgs::ObjectPose>> sorted_data;
    for (size_t i = 0; i < bb_msg->bounding_boxes.size(); ++i) {
        sorted_data.emplace_back(bb_msg->bounding_boxes[i], op_msg->object_poses[i]);
    }

    std::sort(sorted_data.begin(), sorted_data.end(), [](const auto &a, const auto &b) {
        return a.first.xmin < b.first.xmin;
    });

    // ソート済みデータを再配置
    std::vector<sobits_msgs::BoundingBox> sorted_bb;
    std::vector<sobits_msgs::ObjectPose> sorted_op;
    for (const auto &p : sorted_data) {
        sorted_bb.push_back(p.first);
        sorted_op.push_back(p.second);
    }

    // (2) Tracklet 作成
    std::unordered_map<long, SOBITTracklet::Ptr> tracks;
    for (size_t i = 0; i < sorted_op.size(); ++i) {
        tracks[i].reset(new SOBITTracklet(sorted_op[i]));
        tracks[i]->pose_x = sorted_op[i].pose.position.x;
        tracks[i]->pose_y = sorted_op[i].pose.position.y;
    }

    // (3) Bounding Box の処理
    for (size_t i = 0; i < sorted_bb.size(); ++i) {
        int region_x = sorted_bb[i].xmin;
        int region_y = sorted_bb[i].ymin;
        int region_width = sorted_bb[i].xmax - sorted_bb[i].xmin;
        int region_height = sorted_bb[i].ymax - sorted_bb[i].ymin;

        cv::Rect person_region(region_x, region_y, region_width, region_height);
        tracks[i]->person_region = person_region;

        // 画像処理
        cv::Mat extracted_region = img_raw(person_region);
        cv::Mat resized_person_region;
        cv::resize(extracted_region, resized_person_region, cv::Size(128, 320));

        cv_bridge::CvImage img_bridge1;
        sensor_msgs::Image result_img_msg1;
        img_bridge1 = cv_bridge::CvImage(img_msg->header, sensor_msgs::image_encodings::BGR8, resized_person_region);
        img_bridge1.toImageMsg(result_img_msg1);
        image_pub_1.publish(result_img_msg1);
    }
    
    std::lock_guard<std::mutex> lock(context_mutex);
    context->extract_features(img_raw, tracks);
    State* next_state = state->update(nh_, *context, tracks, target_position_keep, target_InitialClassifier_count);
    if(next_state != state.get()) {
        state.reset(next_state);
    }
    // if (target_pub_.getNumSubscribers()){
    SOBITTarget target;
    target.header = img_msg->header;
    target.state.data = state->state_name();
    target.target_id = state->target();

    // ターゲットID取得
    long target_id_keep = state->target();
    if (tracks.find(target_id_keep) != tracks.end()) {
        // ターゲットが存在する場合、x, y 座標値を取得
        target_position_keep.x = tracks[target_id_keep]->pose_x;
        target_position_keep.y = tracks[target_id_keep]->pose_y;

        target.position.x = target_position_keep.x;
        target.position.y = target_position_keep.y;
        
        float distance = std::sqrt(std::pow(target_position_keep.x, 2) + std::pow(target_position_keep.y, 2));
        ROS_INFO("Previous Target ID: %ld, Previous Target Distance: %.2f", target_id_keep, distance);
    } else {
        ROS_WARN("Previous Target ID: %ld not found in tracks", target_id_keep);
    }


    // std::cout << "bb_msg->bounding_boxes.size() : " << bb_msg->bounding_boxes.size() << std::endl;
    target.track_ids.reserve(bb_msg->bounding_boxes.size());
    target.confidences.reserve(bb_msg->bounding_boxes.size());
    // target.classifier_confidences.reserve(bb_msg->bounding_boxes.size() * 2);
    target.classifier_confidences.reserve(bb_msg->bounding_boxes.size());
    std::vector<std::string> classifier_names = context->classifier_names();
    for(const auto& name: classifier_names) {
        std_msgs::String classifier_name;
        classifier_name.data = name;
        target.classifier_names.push_back(classifier_name);
    }
    for(const auto& track : tracks) {
        if(track.second->confidence) {
            target.track_ids.push_back(track.first);
            target.confidences.push_back(*track.second->confidence);

            if(track.second->classifier_confidences.size() != target.classifier_names.size()) {
                ROS_ERROR_STREAM("num_classifiers did not match!!");
                ROS_ERROR_STREAM(track.second->classifier_confidences.size() << " : " << target.classifier_names.size());
            }
            std::copy(track.second->classifier_confidences.begin(), track.second->classifier_confidences.end(), std::back_inserter(target.classifier_confidences));
        }
        // If *track.second->confidence is above 0.01: Same person (red), between -0.1 and 0.3: Reid (green), below -0.1: Different person (blue)
        // FIXME : Adjustment of each parameter
        cv::Scalar color = *track.second->confidence >   0.3 ? cv::Scalar(0, 0, 255) : 
                           *track.second->confidence <  -0.1 ? cv::Scalar(255, 0, 0) : 
                                                                cv::Scalar(0, 255, 0) ;
        cv::rectangle(input_img, *track.second->person_region, color, 2);
        cv::String label = *track.second->confidence >   0.3 ? "same:" + std::to_string(*track.second->confidence): 
                           *track.second->confidence <  -0.1 ? "diff:" + std::to_string(*track.second->confidence):
                                                                 "id:" + std::to_string(*track.second->confidence);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.75, 1.0, 0);

        // cv::Rect labelRect = cv::Rect(cv::Point(track.second->person_region->x, track.second->person_region->y-labelSize.height), cv::Size(labelSize.width, labelSize.height));
        cv::Rect labelRect = cv::Rect(cv::Point(track.second->person_region->x, track.second->person_region->y), cv::Size(labelSize.width, labelSize.height));
        cv::rectangle(input_img, labelRect, cv::Scalar::all(255), -1);
        
        // cv::putText(input_img, label, cv::Point(track.second->person_region->x, track.second->person_region->y), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar::all(0));
        cv::putText(input_img, label, cv::Point(track.second->person_region->x, track.second->person_region->y+labelSize.height), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar::all(0));
        cv::putText(input_img, "state:" + target.state.data, cv::Point(10, 25), cv::FONT_HERSHEY_TRIPLEX, 1.5, cv::Scalar(255, 0, 0), 2);
    }
    target_pub_.publish(target);
    if(pub_target_img){
        cv_bridge::CvImage img_bridge;
        sensor_msgs::Image result_img_msg;
        img_bridge = cv_bridge::CvImage(img_msg->header, sensor_msgs::image_encodings::BGR8, input_img);
        img_bridge.toImageMsg(result_img_msg);
        image_pub_.publish(result_img_msg);
    }
    // }
    
    // if(features_pub.getNumSubscribers()){
    cv::Mat features = context->visualize_body_features();
    if(features.data) {
        cv_bridge::CvImage cv_image(img_msg->header, "bgr8", features);
        features_pub.publish(cv_image.toImageMsg());
    }
    // }
    
    return;
}

void person_id_follow_nodelet::CCFPersonId::reset(long target_id = -1) {
    ROS_INFO_STREAM("reset identification!!");
    std::lock_guard<std::mutex> lock(context_mutex);
    if(target_id < 0) {
        state.reset(new InitialState());
    } else {
        state.reset(new InitialTrainingState(target_id));
    }
    context.reset(new SOBITContextInit(nh_));
}



PLUGINLIB_EXPORT_CLASS(person_id_follow_nodelet::CCFPersonId, nodelet::Nodelet);