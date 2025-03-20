#include "enumerator.hpp"
#include<stack>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

#define DEFAULT_TIMEOUT 1800

PathEnumeratorLabel::PathEnumeratorLabel(size_t node_id) {
    this->node_id = node_id;
    this->hop_count = 0;
}

PathEnumeratorLabel::~PathEnumeratorLabel() = default;

/*
 * Vector computation for edge-labeled patterns.
 */

vector<unsigned long long> PathEnumerator::cardinalities_vector_el_hm(NodeEdgeLabeledGraph &g, vector<string> &pattern) {
    return this->cardinalities_vector_el(g, pattern, HOMOMORPHISM, DEFAULT_TIMEOUT);
}

vector<unsigned long long> PathEnumerator::cardinalities_vector_el_cm(NodeEdgeLabeledGraph &g, vector<string> &pattern) {
    return this->cardinalities_vector_el(g, pattern, CYPHERMORPHISM, DEFAULT_TIMEOUT);
}

vector<unsigned long long> PathEnumerator::cardinalities_vector_el_im(NodeEdgeLabeledGraph &g, vector<string> &pattern) {
    return this->cardinalities_vector_el(g, pattern, ISOMORPHISM, DEFAULT_TIMEOUT);
}

vector<unsigned long long> PathEnumerator::cardinalities_vector_nel_hm(NodeEdgeLabeledGraph &g, vector<string> &pattern) {
    return this->cardinalities_vector_nel(g, pattern, HOMOMORPHISM, DEFAULT_TIMEOUT);
}

vector<unsigned long long> PathEnumerator::cardinalities_vector_nel_cm(NodeEdgeLabeledGraph &g, vector<string> &pattern) {
    return this->cardinalities_vector_nel(g, pattern, CYPHERMORPHISM, DEFAULT_TIMEOUT);
}

vector<unsigned long long> PathEnumerator::cardinalities_vector_nel_im(NodeEdgeLabeledGraph &g, vector<string> &pattern) {
    return this->cardinalities_vector_nel(g, pattern, ISOMORPHISM, DEFAULT_TIMEOUT);
}

vector<unsigned long long> PathEnumerator::cardinalities_vector_el(NodeEdgeLabeledGraph &g, vector<string> &pattern, Semantics sem, int timeout) {
    vector<unsigned long long> path_count_vector(pattern.size(),0);
    auto time_start = high_resolution_clock::now();

    std::stack<PathEnumeratorLabel*> dfs_stack;

    std::vector<PathEnumeratorLabel*> all_created_labels;

    NodeEdgeLabeledGraph::vertex_iterator v, vend;
    typename graph_traits < NodeEdgeLabeledGraph >::out_edge_iterator ei, ei_end;

    boost::property_map< NodeEdgeLabeledGraph, edge_index_t >::type edge_index_map = boost::get(boost::edge_index, g);
    boost::property_map< NodeEdgeLabeledGraph, edge_name_t >::type edge_name_map = boost::get(boost::edge_name, g);

    for (boost::tie(v, vend) = vertices(g); v != vend; ++v) {

        auto time_check = high_resolution_clock::now();
        duration<double, std::milli> ms_double_pre = time_check - time_start;
        if(ms_double_pre.count()/1000 > timeout)
            return vector<unsigned long long>(pattern.size(),0);

        auto *pel = new PathEnumeratorLabel(*v);
        pel->hop_count = 0;
        dfs_stack.push(pel);
        all_created_labels.push_back(pel);

        // Expanding DFS search
        while(!dfs_stack.empty()) {
            PathEnumeratorLabel *current_pel = dfs_stack.top();
            dfs_stack.pop();

            std::unordered_set<size_t> ids;
            if(sem == CYPHERMORPHISM || sem == ISOMORPHISM) {
                PathEnumeratorLabel *temp = current_pel;
                while(temp != NULL) {
                    ids.insert(temp->previous_id);
                    temp = temp->previous_pel;
                }
            }

            for (boost::tie(ei, ei_end) = boost::out_edges(current_pel->node_id, g); ei != ei_end; ++ei) {
                if(!(edge_name_map[*ei] == pattern[current_pel->hop_count]))
                    continue;
                if(sem == CYPHERMORPHISM) {
                    if(ids.contains(edge_index_map[*ei]))
                        continue;
                }
                else if(sem == ISOMORPHISM) {
                    if(ids.contains(boost::target(*ei,g)))
                        continue;
                }
                path_count_vector[current_pel->hop_count] += 1;

                if (current_pel->hop_count+1 < pattern.size()) {
                    auto *next_pel = new PathEnumeratorLabel((*ei).m_target);
                    next_pel->hop_count = current_pel->hop_count + 1;
                    if(sem == CYPHERMORPHISM) {
                        next_pel->previous_pel = current_pel;
                        next_pel->previous_id = edge_index_map[*ei];
                    }
                    else if(sem == ISOMORPHISM) {
                        next_pel->previous_pel = current_pel;
                        next_pel->previous_id = current_pel->node_id;
                    }
                    dfs_stack.push(next_pel);
                    all_created_labels.push_back(next_pel);
                }
            }
        }

        // Clearing labels
        for(auto & all_created_label : all_created_labels)
            delete all_created_label;
        all_created_labels.clear();
    }
    return path_count_vector;
}

/*
 * Vector computation for node and edge-labeled patterns.
 */

vector<unsigned long long> PathEnumerator::cardinalities_vector_nel(NodeEdgeLabeledGraph &g, PathPattern &pattern, Semantics sem, int timeout) {
    vector<unsigned long long> path_count_vector(pattern.size(),0);
    auto time_start = high_resolution_clock::now();

    std::stack<PathEnumeratorLabel*> dfs_stack;

    std::vector<PathEnumeratorLabel*> all_created_labels;

    NodeEdgeLabeledGraph::vertex_iterator v, vend;
    typename graph_traits < NodeEdgeLabeledGraph >::out_edge_iterator ei, ei_end;

    boost::property_map< NodeEdgeLabeledGraph, edge_index_t >::type edge_index_map = boost::get(boost::edge_index, g);
    boost::property_map< NodeEdgeLabeledGraph, edge_name_t >::type edge_name_map = boost::get(boost::edge_name, g);
    boost::property_map< NodeEdgeLabeledGraph, vertex_name_t >::type vertex_name_map = boost::get(boost::vertex_name, g);

    for (boost::tie(v, vend) = vertices(g); v != vend; ++v) {

        auto time_check = high_resolution_clock::now();
        duration<double, std::milli> ms_double_pre = time_check - time_start;
        if(ms_double_pre.count()/1000 > timeout)
            return vector<unsigned long long>(pattern.size(),0);

        if(pattern.at(0) != vertex_name_map[*v] && pattern.at(0) != "_")
            continue;

        auto *pel = new PathEnumeratorLabel(*v);
        pel->hop_count = 1;
        path_count_vector[0] += 1;
        dfs_stack.push(pel);
        all_created_labels.push_back(pel);

        // Expanding DFS search
        while(!dfs_stack.empty()) {
            PathEnumeratorLabel *current_pel = dfs_stack.top();
            dfs_stack.pop();

            std::unordered_set<size_t> ids;
            if(sem == CYPHERMORPHISM || sem == ISOMORPHISM) {
                PathEnumeratorLabel *temp = current_pel;
                while(temp != NULL) {
                    ids.insert(temp->previous_id);
                    temp = temp->previous_pel;
                }
            }

            for (boost::tie(ei, ei_end) = boost::out_edges(current_pel->node_id, g); ei != ei_end; ++ei) {

            
                if(!(edge_name_map[*ei] == pattern[current_pel->hop_count]) && !(pattern[current_pel->hop_count] == "_"))
                    continue;

                if(sem == CYPHERMORPHISM) {
                    if(ids.contains(edge_index_map[*ei]))
                        continue;
                }
                else if(sem == ISOMORPHISM) {
                    if(ids.contains(boost::target(*ei,g)))
                        continue;
                }

                path_count_vector[current_pel->hop_count] += 1;
                if(!(vertex_name_map[boost::target(*ei,g)] == pattern[current_pel->hop_count+1]) && !(pattern[current_pel->hop_count+1] == "_"))
                    continue;
                path_count_vector[current_pel->hop_count+1] += 1;

                if (current_pel->hop_count+1 < pattern.size()) {
                    auto *next_pel = new PathEnumeratorLabel((*ei).m_target);
                    next_pel->hop_count = current_pel->hop_count + 2;

                    if(sem == CYPHERMORPHISM) {
                        next_pel->previous_pel = current_pel;
                        next_pel->previous_id = edge_index_map[*ei];
                    }
                    else if(sem == ISOMORPHISM) {
                        next_pel->previous_pel = current_pel;
                        next_pel->previous_id = current_pel->node_id;
                    }

                    dfs_stack.push(next_pel);
                    all_created_labels.push_back(next_pel);
                }
            }
        }

        // Clearing labels
        for(auto & all_created_label : all_created_labels)
            delete all_created_label;
        all_created_labels.clear();
    }
    return path_count_vector;
}
