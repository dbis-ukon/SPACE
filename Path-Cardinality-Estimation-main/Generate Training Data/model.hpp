#ifndef PATHENUMERATION_MODEL_HPP
#define PATHENUMERATION_MODEL_HPP

#include <iostream>
#include <string>
#include <boost/graph/adjacency_list.hpp>

typedef std::vector<std::string> PathPattern;
typedef std::vector<std::string> PGPathPattern;

using namespace boost;

typedef adjacency_list<
    vecS,
    vecS,
    directedS,
    property<vertex_index_t, std::size_t, property<boost::vertex_name_t, std::string>>,
    property<edge_index_t, std::size_t, property<boost::edge_name_t, std::string>>,
    no_property,
    vecS
> NodeEdgeLabeledGraph;

typedef graph_traits<NodeEdgeLabeledGraph>::vertex_descriptor Node;
typedef graph_traits<NodeEdgeLabeledGraph>::edge_descriptor Relationship;

NodeEdgeLabeledGraph load_edge_labeled_graph(const char *filename);
NodeEdgeLabeledGraph load_node_edge_labeled_graph(const char *edges_file, const char *node_labels_file);
std::unordered_map<std::string,int> get_edge_label_distribution(NodeEdgeLabeledGraph &g);
std::vector<std::string> get_edge_labels(NodeEdgeLabeledGraph &g);

#endif
