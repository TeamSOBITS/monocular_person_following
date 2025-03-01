#ifndef REID_STATE_HPP
#define REID_STATE_HPP

#include <person_id_follow/state/state.hpp>

#include <geometry_msgs/Point.h>

namespace person_id_follow {

class ReidState : public State {
public:
    ReidState() {}
    virtual ~ReidState() override {}

    virtual std::string state_name() const override {
        return "re-identification";
    }

    virtual State* update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) override;

private:
    std::unordered_map<long, int> positive_count;
};

}

#endif // REID_STATE_HPP
