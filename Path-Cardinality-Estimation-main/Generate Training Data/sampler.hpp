#ifndef PATHENUMERATION_SAMPLER_HPP
#define PATHENUMERATION_SAMPLER_HPP

#include "model.hpp"

class PatternSampler {
public:
    std::vector<PathPattern> random_walk_sampler(NodeEdgeLabeledGraph &g, unsigned int pattern_count, unsigned int max_pattern_size);
    std::vector<PathPattern> random_walk_sampler_2(NodeEdgeLabeledGraph &g, unsigned int pattern_count, unsigned int max_pattern_size);
};

#endif
