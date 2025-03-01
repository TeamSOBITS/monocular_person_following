#ifndef SOBIT_CONTEXT_INIT_HPP
#define SOBIT_CONTEXT_INIT_HPP

#include <random>
#include <unordered_map>
#include <boost/optional.hpp>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <person_id_follow/sobit_tracklet.hpp>
#include <ccf_person_identification/sobit_person_classifier.hpp>

namespace ccf_person_classifier {

class PersonInput;
class PersonFeatures;
class PersonClassifier;

}


namespace person_id_follow {

class SOBITContextInit {
public:
    using PersonInput = ccf_person_classifier::PersonInput;
    using PersonFeatures = ccf_person_classifier::PersonFeatures;
    using PersonClassifier = ccf_person_classifier::PersonClassifier;

    SOBITContextInit(ros::NodeHandle& nh);
    ~SOBITContextInit();

public:
    ccf_person_classifier::Features::Ptr extract_features_init(std::unordered_map<std::string, cv::Mat>& image);
    void extract_features(const cv::Mat& bgr_image, std::unordered_map<long, SOBITTracklet::Ptr>& tracks);
    void update_classifier_init(double label, const ccf_person_classifier::Features::Ptr& features);
    bool update_classifier(double label, long track_num, const SOBITTracklet::Ptr& track);
    boost::optional<double> predict_init(const ccf_person_classifier::Features::Ptr& features);
    boost::optional<double> predict(long track_num, const SOBITTracklet::Ptr& track);
    std::vector<std::string> classifier_names() const;
    cv::Mat visualize_body_features();

private:
    std::mt19937 mt;
    std::unique_ptr<PersonClassifier> classifier;
    // std::shared_ptr<PersonClassifier> classifier;
    // std::shared_ptr<PersonClassifier> classifier_init;

    std::unordered_map<long, std::vector<double>> classifier_confidences;
    // std::shared_ptr<std::unordered_map<long, std::vector<double>>> classifier_confidences_init;
    ccf_person_classifier::Input::Ptr input;
    ccf_person_classifier::Features::Ptr features;

    boost::circular_buffer<std::shared_ptr<ccf_person_classifier::Features>> pos_feature_bank;
    boost::circular_buffer<std::shared_ptr<ccf_person_classifier::Features>> neg_feature_bank;
};

}

#endif //SOBIT_CONTEXT_HPP
