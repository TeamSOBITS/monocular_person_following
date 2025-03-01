
#include <cv_bridge/cv_bridge.h>
#include "person_id_follow/sobit_context_init.hpp"
#include "ccf_person_identification/sobit_person_classifier.hpp"

namespace person_id_follow {

SOBITContextInit::SOBITContextInit(ros::NodeHandle& nh) {
    classifier.reset(new PersonClassifier(nh));
    // classifier = std::make_shared<PersonClassifier>(nh);
    // classifier_init = classifier;

    pos_feature_bank.resize(nh.param<int>("feature_bank_size", 32));
    neg_feature_bank.resize(nh.param<int>("feature_bank_size", 32));
}

SOBITContextInit::~SOBITContextInit() {}

ccf_person_classifier::Features::Ptr SOBITContextInit::extract_features_init(std::unordered_map<std::string, cv::Mat>& image){
    ccf_person_classifier::Input::Ptr input(new PersonInput());
    ccf_person_classifier::Features::Ptr features(new PersonFeatures());
    // input.reset(new ccf_person_classifier::PersonInput());
    // features.reset(new ccf_person_classifier::PersonFeatures());
    if(!classifier->extractInput(input, image)) {
        ROS_WARN_STREAM("failed to extract input data");
    }
    if(!classifier->extractFeatures(features, input)) {
        ROS_WARN_STREAM("failed to extract input features");
    }
    return features;
}

void SOBITContextInit::extract_features(const cv::Mat& bgr_image, std::unordered_map<long, SOBITTracklet::Ptr>& tracks) {
    for(auto& track: tracks) {
        if(!track.second->person_region) {
            std::cout << "!track.second->person_region" << std::endl;
            continue;
        }
        if(track.second->person_region->width < 20 || track.second->person_region->height < 20) {
            std::cout << "track.second->person_region->width < 20 || track.second->person_region->height < 20" << std::endl;
            continue;
        }
        
        track.second->input.reset(new PersonInput());
        track.second->features.reset(new PersonFeatures());
        
        std::unordered_map<std::string, cv::Mat> image;
        image["body"] = cv::Mat(bgr_image, *track.second->person_region);

        // FIXME:Resize adjustment for each camera
        // cv::resize(image["body"], image["body"],cv::Size(128, 256));

        // cv::resize(image["body"], image["body"],cv::Size(128, 340));//リサイズ比1:2.66（論文の入力画像サイズ比に基づいて）
        cv::resize(image["body"], image["body"],cv::Size(128, 320));//リサイズ比（MPF_GRR_SLT論文の手法で用いる入力画像サイズ比に基づいて）

        // image["face"] = track.second->face_image;
        if(!classifier->extractInput(track.second->input, image)) {
            ROS_WARN_STREAM("failed to extract input data");
            continue;
        }
        if(!classifier->extractFeatures(track.second->features, track.second->input)) {
            ROS_WARN_STREAM("failed to extract input features");
            continue;
        }
    }
}
void SOBITContextInit::update_classifier_init(double label, const ccf_person_classifier::Features::Ptr& features) {
    boost::optional<double> pred = *classifier->predict(features, classifier_confidences[0]);
    std::cout << "pred : " << *classifier->predict(features, classifier_confidences[0]) << std::endl;
    auto& p_bank = label > 0.0 ? pos_feature_bank : neg_feature_bank;
    auto& n_bank = label > 0.0 ? neg_feature_bank : pos_feature_bank;
    
    if(!n_bank.empty()) {
        size_t i = std::uniform_int_distribution<>(0, n_bank.size())(mt);
        // classifier_init->update(-label, n_bank[i]);
        classifier->update(-label, n_bank[i]);
    }
    if(!p_bank.full()) {
        p_bank.push_back(features);
    } else {
        size_t i = std::uniform_int_distribution<>(0, p_bank.size())(mt);
        std::cout << "i : " << i << std::endl;
        p_bank[i] = features;
    }
    // classifier_init->update(label, features);
    classifier->update(label, features);
}


// 修正by mukogawa
bool SOBITContextInit::update_classifier(double label, long track_num, const SOBITTracklet::Ptr& track) {
    // boost::optional<double> pred = *classifier->predict(track->features, classifier_confidences[(track->op_msg->detect_id) - 1]);
    boost::optional<double> pred = *classifier->predict(track->features, classifier_confidences[track_num]);
    // std::cout << "pred : " << *classifier->predict(track->features, classifier_confidences[(track->op_msg->detect_id) - 1]) << std::endl;
    std::cout << "pred : " << *pred << std::endl;
    std::cout << "label : " << label << std::endl;
    std::cout << "track->op_msg->detect_id : " << track->op_msg->detect_id << std::endl;
    std::cout << "track_num : " << track_num << std::endl;
    // auto pred = classifier->predict(track->features);
    if(pred) {
        track->confidence = pred;
    }
    // track->classifier_confidences = classifier_confidences[(track->op_msg->detect_id) - 1];
    track->classifier_confidences = classifier_confidences[track_num];
    auto& p_bank = label > 0.0 ? pos_feature_bank : neg_feature_bank;
    auto& n_bank = label > 0.0 ? neg_feature_bank : pos_feature_bank;
    
    if(!n_bank.empty()) {
        size_t i = std::uniform_int_distribution<>(0, n_bank.size())(mt);
        classifier->update(-label, n_bank[i]);
    }
    if(!p_bank.full()) {
        p_bank.push_back(track->features);
    } else {
        size_t i = std::uniform_int_distribution<>(0, p_bank.size())(mt);
        std::cout << "i : " << i << std::endl;
        p_bank[i] = track->features;
    }
    return classifier->update(label, track->features);
}

// 修正by mukogawa
boost::optional<double> SOBITContextInit::predict(long track_num, const SOBITTracklet::Ptr& track) {
    // classifier->predict: return confidence level based on features (online_boosting.hpp: predictReal)
    // Refer to ccf_person_identification_initial.cpp
    // boost::optional<double> pred = *classifier->predict(track->features, classifier_confidences[(track->op_msg->detect_id) - 1]);
    boost::optional<double> pred = *classifier->predict(track->features, classifier_confidences[track_num]);
    if(pred) {
        // std::cout << "pred success" <<std::endl;
        track->confidence = pred;
    }
    // std::cout << "predict-pred : " << *pred << std::endl;
    // std::cout << "predict-track->op_msg->detect_id : " << track->op_msg->detect_id - 1 << std::endl;
    // std::cout << "predict-track_num : " << track_num << std::endl;
    
    // track->classifier_confidences = classifier_confidences[(track->op_msg->detect_id) - 1];
    track->classifier_confidences = classifier_confidences[track_num];
    return pred;
}

boost::optional<double> SOBITContextInit::predict_init(const ccf_person_classifier::Features::Ptr& features) {
    std::cout << "predict_init" << std::endl;
    boost::optional<double> pred = *classifier->predict(features, classifier_confidences[0]);
    return pred;
}

std::vector<std::string> SOBITContextInit::classifier_names() const {
    return classifier->classifierNames();
}


cv::Mat SOBITContextInit::visualize_body_features() {
    ccf_person_classifier::BodyClassifier::Ptr body_classifier = classifier->getClassifier<ccf_person_classifier::BodyClassifier>("body");
    if(body_classifier) {
        cv::Mat feature_map = body_classifier->visualize();
        return feature_map;
    }

    return cv::Mat();
}


}


