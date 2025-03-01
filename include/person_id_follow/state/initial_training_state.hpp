#ifndef INITIAL_TRAINING_STATE_HPP
#define INITIAL_TRAINING_STATE_HPP

#include <person_id_follow/state/state.hpp>

#include <geometry_msgs/Point.h>

namespace person_id_follow {

class InitialTrainingState : public State {
public:
    InitialTrainingState(long target_id)
        : target_id(target_id),
          num_pos_samples(0)
    {}

    virtual ~InitialTrainingState() override {}

    virtual long target() const override {
        return target_id;
    }

    virtual std::string state_name() const override {
        return "initial_training";
    }

    virtual State* update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks, geometry_msgs::Point& target_position_keep, int& target_InitialClassifier_count) override;

private:
    long target_id;
    long num_pos_samples;
    long lost_count;
};

}

#endif // INITIAL_TRAINING_STATE_HPP
