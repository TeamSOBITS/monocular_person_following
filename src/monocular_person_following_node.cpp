#include <mutex>
#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>

#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>

#include <monocular_person_following/Target.h>
#include <monocular_person_following/Imprint.h>
// #include <monocular_person_following/FaceDetectionArray.h>
#include <monocular_people_tracking/TrackArray.h>

#include <monocular_person_following/context.hpp>
#include <monocular_person_following/tracklet.hpp>
#include <monocular_person_following/state/state.hpp>
#include <monocular_person_following/state/initial_state.hpp>
#include <monocular_person_following/state/initial_training_state.hpp>


namespace monocular_person_following {

class MonocularPersonFollowingNode {
public:
    MonocularPersonFollowingNode()
        : nh(),
          private_nh("~"),
          target_pub(nh.advertise<Target>("/monocular_person_following/target", 1)),
          image_trans(nh),
          features_pub(image_trans.advertise("/monocular_person_following/features", 1)),
          image_sub(nh, "image", 10),
          tracks_sub(nh, "/monocular_people_tracking/tracks", 10),
        //   faces_sub(nh, "/face_detector/faces", 10),
          sync(image_sub, tracks_sub, 30),
        //   sync_w_face(image_sub, tracks_sub, faces_sub, 30),
          reset_sub(private_nh.subscribe<std_msgs::Empty>("reset", 10, &MonocularPersonFollowingNode::reset_callback, this)),
          reset_service_server(private_nh.advertiseService("reset", &MonocularPersonFollowingNode::reset_service, this)),
          imprint_service_server(private_nh.advertiseService("imprint", &MonocularPersonFollowingNode::imprint_service, this))
    {
        state.reset(new InitialState());
        context.reset(new Context(private_nh));

        sync.registerCallback(boost::bind(&MonocularPersonFollowingNode::callback, this, _1, _2));
        // if(private_nh.param<bool>("use_face", true)) {
        //     sync_w_face.registerCallback(boost::bind(&MonocularPersonFollowingNode::callback, this, _1, _2, _3));
        // } else {
        //     sync.registerCallback(boost::bind(&MonocularPersonFollowingNode::callback, this, _1, _2, nullptr));
        // }
    }

    // void callback(const sensor_msgs::ImageConstPtr& image_msg, const monocular_people_tracking::TrackArrayConstPtr& tracks_msg, const monocular_person_following::FaceDetectionArrayConstPtr& faces_msg) {
    void callback(const sensor_msgs::ImageConstPtr& image_msg, const monocular_people_tracking::TrackArrayConstPtr& tracks_msg) {
        auto cv_image = cv_bridge::toCvCopy(image_msg, "bgr8");

        std::unordered_map<long, Tracklet::Ptr> tracks;

        // std::unordered_map<long, FaceDetection const*> face_msgs;
        // if(faces_msg != nullptr) {
        //     for(const auto& face : faces_msg->faces) {
        //         face_msgs[face.track_id] = &face;
        //     }
        // }

        for(const auto& track : tracks_msg->tracks) {
            tracks[track.id].reset(new Tracklet(tf_listener, tracks_msg->header, track));

            if(track.associated_neck_ankle.empty()) {
                continue;
            }

            cv::Rect person_region = calc_person_region(track, cv_image->image.size());
            tracks[track.id]->person_region = person_region;

            // auto face = face_msgs.find(track.id);
            // if(face != face_msgs.end() && !face->second->face_image.empty()) {
            //     auto face_image = cv_bridge::toCvCopy(face->second->face_image[0], "bgr8");
            //     tracks[track.id]->face_image = face_image->image;
            // }
        }

        std::lock_guard<std::mutex> lock(context_mutex);
        context->extract_features(cv_image->image, tracks);

        State* next_state = state->update(private_nh, *context, tracks);
        if(next_state != state.get()) {
            state.reset(next_state);
        }

        if(target_pub.getNumSubscribers()) {
            Target target;
            target.header = image_msg->header;
            target.state.data = state->state_name();
            target.target_id = state->target();

            target.track_ids.reserve(tracks_msg->tracks.size());
            target.confidences.reserve(tracks_msg->tracks.size());
            target.classifier_confidences.reserve(tracks_msg->tracks.size() * 2);

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
            }

            target_pub.publish(target);
        }

        if(features_pub.getNumSubscribers()) {
            cv::Mat features = context->visualize_body_features();
            if(features.data) {
                cv_bridge::CvImage cv_image(image_msg->header, "bgr8", features);
                features_pub.publish(cv_image.toImageMsg());
            }
        }
    }

    bool imprint_service(ImprintRequest& req, ImprintResponse& res) {
        reset(req.target_id);
        res.success = true;

        return true;
    }

    bool reset_service(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
        reset();

        return true;
    }

    void reset_callback(const std_msgs::EmptyConstPtr& empty_msg) {
        reset();
    }

private:
    cv::Rect2f calc_person_region(const monocular_people_tracking::Track& track, const cv::Size& image_size) {
        Eigen::Vector2f neck(track.associated_neck_ankle[0].x, track.associated_neck_ankle[0].y);
        Eigen::Vector2f ankle(track.associated_neck_ankle[1].x, track.associated_neck_ankle[1].y);

        Eigen::Vector2f center = (neck + ankle) / 2.0f;
        float height = (ankle.y() - neck.y()) * 1.25f;
        float width = height * 0.25f;

        cv::Rect rect(center.x() - width / 2.0f, center.y() - height / 2.0f, width, height);

        cv::Point tl = rect.tl();
        cv::Point br = rect.br();

        tl.x = std::min(image_size.width, std::max(0, tl.x));
        tl.y = std::min(image_size.height, std::max(0, tl.y));
        br.x = std::min(image_size.width, std::max(0, br.x));
        br.y = std::min(image_size.height, std::max(0, br.y));

        return cv::Rect(tl, br);
    }

    void reset(long target_id = -1) {
        ROS_INFO_STREAM("reset identification!!");
        std::lock_guard<std::mutex> lock(context_mutex);

        if(target_id < 0) {
            state.reset(new InitialState());
        } else {
            state.reset(new InitialTrainingState(target_id));
        }
        context.reset(new Context(private_nh));
    }

private:
    ros::NodeHandle nh;
    ros::NodeHandle private_nh;

    tf::TransformListener tf_listener;

    ros::Publisher target_pub;

    image_transport::ImageTransport image_trans;
    image_transport::Publisher features_pub;

    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    message_filters::Subscriber<monocular_people_tracking::TrackArray> tracks_sub;
    // message_filters::Subscriber<monocular_person_following::FaceDetectionArray> faces_sub;
    message_filters::TimeSynchronizer<sensor_msgs::Image, monocular_people_tracking::TrackArray> sync;
    // message_filters::TimeSynchronizer<sensor_msgs::Image, monocular_people_tracking::TrackArray, monocular_person_following::FaceDetectionArray> sync_w_face;

    // reset service callbacks
    ros::Subscriber reset_sub;
    ros::ServiceServer reset_service_server;
    ros::ServiceServer imprint_service_server;

    std::mutex context_mutex;
    std::shared_ptr<State> state;
    std::unique_ptr<Context> context;
};

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "monocular_person_following");
    std::unique_ptr<monocular_person_following::MonocularPersonFollowingNode> node(new monocular_person_following::MonocularPersonFollowingNode());
    ros::spin();

    return 0;
}
