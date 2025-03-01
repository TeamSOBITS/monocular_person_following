#ifndef SOBIT_TRACKLET_HPP
#define SOBIT_TRACKLET_HPP
#include <vector>
#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>
#include <ccf_person_identification/online_classifier.hpp>
#include <sobits_msgs/ObjectPose.h>
namespace person_id_follow {

struct SOBITTracklet {
public:
    using Ptr = std::shared_ptr<SOBITTracklet>;
    SOBITTracklet(const sobits_msgs::ObjectPose& op_msg)
        : op_msg(&op_msg)
    {}

public:
    boost::optional<double> confidence;
    std::vector<double> classifier_confidences;

    // cv::Mat face_image;

    boost::optional<cv::Rect> person_region;
    ccf_person_classifier::Input::Ptr input;
    ccf_person_classifier::Features::Ptr features;
    
    float pose_x;
    float pose_y;

    // int idx;
    const sobits_msgs::ObjectPose* op_msg;
    
};

}

#endif // TRACKLET_HPP
