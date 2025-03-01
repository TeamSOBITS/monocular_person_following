#include "person_id_follow/state/init_state.hpp"

#include "person_id_follow/state/initial_state.hpp"

#include "ccf_person_identification/sobit_person_classifier.hpp"

#include <geometry_msgs/Point.h>

namespace person_id_follow {

// If the subject's image can be prepared in advance, init_state can be used.
State* InitState::update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count){
    std::cout << "---------------------" << std::endl;
    std::cout << "\033[1;33mInitState \033[0mstart!" << std::endl;
    std::cout << "---------------------" << std::endl;

    // Specify the path of the image you want to load
    std::string dataset_dir = "/home/sobits/catkin_ws/src/person_id_follow/ccf_person_identification/data/test";
    double label = -1.0;
    boost::optional<double> pred = 0.0;

    for(int i=1; i<=14; i++) {
        std::cout << i << "回目" << std::endl;
        ccf_person_classifier::Features::Ptr features(new ccf_person_classifier::PersonFeatures);
        std::unordered_map<std::string, cv::Mat> pos, neg1, neg2;
        //pos["body"], neg1["body"], neg2["body"] reads the image
        pos["body"] = cv::imread((boost::format("%s/p%02d.jpg") % dataset_dir % i).str());
        neg1["body"] = cv::imread((boost::format("%s/n%02d-01.jpg") % dataset_dir % i).str());
        neg2["body"] = cv::imread((boost::format("%s/n%02d-02.jpg") % dataset_dir % i).str());
        if(!pos["body"].data || !neg1["body"].data || !neg2["body"].data) {
            std::cout << "\033[1;31merror : failed to open image!! image_id: " << i << std::endl;
            return this;
        }
        
        cv::resize(pos["body"], pos["body"], cv::Size(128, 256));

        // cv::Mat pos_result;
        // cv::resize(pos["body"], pos_result, cv::Size(128, 256));
        
        // cv::Mat neg1_result;
        // cv::resize(neg1["body"], neg1_result, cv::Size(128, 256));
        
        // cv::Mat neg2_result;
        // cv::resize(neg2["body"], neg2_result, cv::Size(128, 256));

        features = context.extract_features_init(pos);
        pred = context.predict_init(features);
        context.update_classifier_init(label, features);        
        
        // features = context.extract_features_init(neg1);
        // pred = context.predict_init(features);
        // context.update_classifier_init(label, features);
        
        // features = context.extract_features_init(neg2);
        // pred = context.predict_init(features);
        // context.update_classifier_init(label, features);

    }
    return new InitialState();
}

}