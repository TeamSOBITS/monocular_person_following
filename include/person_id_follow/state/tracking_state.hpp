#ifndef TRACKING_STATE_HPP
#define TRACKING_STATE_HPP

#include <person_id_follow/state/state.hpp>

#include <geometry_msgs/Point.h>

namespace person_id_follow {

class TrackingState : public State {
public:
    TrackingState(long target_id)
        : target_id(target_id)
    {}

    virtual ~TrackingState() override {}

    virtual long target() const override {
        return target_id;
    }

    virtual std::string state_name() const override {
        return "tracking";
    }

    virtual State* update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) override;

private:
    long target_id;
};

}

#endif // TRACKING_STATE_HPP
