#ifndef INITIAL_STATE_HPP
#define INITIAL_STATE_HPP

#include <person_id_follow/state/state.hpp>

#include <geometry_msgs/Point.h>

namespace person_id_follow {

class InitialState : public State {
public:
    InitialState() {}

    virtual ~InitialState() override {}

    virtual std::string state_name() const override {
        return "initial";
    }

    virtual State* update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) override;

private:
    long select_target(ros::NodeHandle& nh, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks);
};

}

#endif // INITIAL_STATE_HPP
