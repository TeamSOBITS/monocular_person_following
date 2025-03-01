#ifndef PERSON_ID_FOLLOW_STATE_INIT_HPP
#define PERSON_ID_FOLLOW_STATE_INIT_HPP

// #include <person_id_follow/sobit_context.hpp>
#include <person_id_follow/sobit_context_init.hpp>
namespace person_id_follow {

class State {
public:
    State() {}
    virtual ~State() {}

    virtual long target() const { return -1; }

    virtual std::string state_name() const = 0;

    virtual State* update(ros::NodeHandle& nh, SOBITContextInit& context, const std::unordered_map<long, SOBITTracklet::Ptr>& tracks) = 0;
private:

};

}

#endif
