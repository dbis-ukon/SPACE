#include "sampler.hpp"

#define MAX_NUMBER_OF_WALKS 10000000

using namespace std;

vector<PathPattern> PatternSampler::random_walk_sampler(NodeEdgeLabeledGraph &g, unsigned int pattern_count, unsigned int max_pattern_size) {
    vector<PathPattern> patterns;
    std::unordered_set<vector<string>, boost::hash<vector<string>>> pattern_set;

    srand( (unsigned)time(NULL) );
    NodeEdgeLabeledGraph::vertex_iterator v, vend;
    typename graph_traits < NodeEdgeLabeledGraph >::out_edge_iterator ei, ei_end;

    boost::property_map< NodeEdgeLabeledGraph, edge_index_t >::type edge_index_map = boost::get(boost::edge_index, g);
    boost::property_map< NodeEdgeLabeledGraph, edge_name_t >::type edge_name_map = boost::get(boost::edge_name, g);

    // Determining starting node.
    int loop_count = 0;
    while(pattern_set.size() < pattern_count) {
        if(loop_count++ == MAX_NUMBER_OF_WALKS) {
            cerr << loop_count << " " << pattern_set.size() << endl;
            break;
        }
        vector<string> pattern;
        int starting_node_idx = std::rand() % boost::num_vertices(g);
        boost::tie(v, vend) = vertices(g);
        v = v + starting_node_idx;
        auto current_node = *v;

        int targeted_pattern_size = (std::rand() % (max_pattern_size-1))+2;
        //cout << targeted_pattern_size;

        while (pattern.size() < targeted_pattern_size) {
            vector<Relationship> next_nodes;
            for (boost::tie(ei, ei_end) = boost::out_edges(current_node, g); ei != ei_end; ++ei) {
                auto next_node = boost::target(*ei, g);
                next_nodes.push_back(*ei);
            }
            if (next_nodes.size() == 0)
                break;
            Relationship rel = next_nodes.at(std::rand() % next_nodes.size());
            pattern.push_back(edge_name_map[rel]);
            current_node = boost::target(rel, g);
        }
        //cout << " - " << pattern.size();
        if(pattern.size() == targeted_pattern_size && !pattern_set.contains(pattern)) {
            pattern_set.insert(pattern);
            //cout << " - " << "SUCCESS" << endl;
        }
        else {
            //cout << endl;
        }
        pattern.clear();
    }
    for(auto it : pattern_set)
        patterns.push_back(it);
    return patterns;
}

std::vector<PathPattern> PatternSampler::random_walk_sampler_2(NodeEdgeLabeledGraph &g, unsigned int pattern_count, unsigned int max_hops) {
    vector<PathPattern> patterns;
    std::unordered_set<vector<string>, boost::hash<vector<string>>> pattern_set;

    srand( (unsigned)time(NULL) );
    NodeEdgeLabeledGraph::vertex_iterator v, vend;
    typename graph_traits < NodeEdgeLabeledGraph >::out_edge_iterator ei, ei_end;

    boost::property_map< NodeEdgeLabeledGraph, vertex_name_t >::type vertex_name_map = boost::get(boost::vertex_name, g);

    boost::property_map< NodeEdgeLabeledGraph, edge_index_t >::type edge_index_map = boost::get(boost::edge_index, g);
    boost::property_map< NodeEdgeLabeledGraph, edge_name_t >::type edge_name_map = boost::get(boost::edge_name, g);

    // Determining starting node.
    int loop_count = 0;
    while(pattern_set.size() < pattern_count) {
        if(loop_count++ == MAX_NUMBER_OF_WALKS) {
            cerr << loop_count << " " << pattern_set.size() << endl;
            break;
        }
        vector<string> pattern;
        int starting_node_idx = std::rand() % boost::num_vertices(g);
        pattern.push_back(vertex_name_map[starting_node_idx]);

        boost::tie(v, vend) = vertices(g);
        v = v + starting_node_idx;
        auto current_node = *v;

        int targeted_pattern_size = (std::rand() % (max_hops-1))+2;
        //cout << targeted_pattern_size;

        while ((pattern.size()-1)/2 < targeted_pattern_size) {
            vector<Relationship> next_nodes;
            for (boost::tie(ei, ei_end) = boost::out_edges(current_node, g); ei != ei_end; ++ei) {
                auto next_node = boost::target(*ei, g);
                next_nodes.push_back(*ei);
            }
            if (next_nodes.size() == 0)
                break;
            Relationship rel = next_nodes.at(std::rand() % next_nodes.size());
            pattern.push_back(edge_name_map[rel]);
            current_node = boost::target(rel, g);
            pattern.push_back(vertex_name_map[current_node]);
        }
        //cout << " - " << pattern.size();
        if((pattern.size()-1)/2 == targeted_pattern_size && !pattern_set.contains(pattern)) {
            pattern_set.insert(pattern);
            //cout << " - " << "SUCCESS" << endl;
        }
        else {
            //cout << endl;
        }
        pattern.clear();
    }
    for(auto it : pattern_set)
        patterns.push_back(it);
    return patterns;
}