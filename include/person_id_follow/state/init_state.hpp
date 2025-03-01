#ifndef INIT_STATE_HPP
#define INIT_STATE_HPP

#include <person_id_follow/state/state.hpp>

#include <geometry_msgs/Point.h>

namespace person_id_follow {

class InitState : public State {
public:
    InitState()
    {}

    virtual ~InitState() override {}

    virtual std::string state_name() const override {
        return "init";
    }

    virtual State* update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) override;

};

}

#endif // INIT_STATE_HPP
