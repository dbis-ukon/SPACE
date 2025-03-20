#include "model.hpp"
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;

NodeEdgeLabeledGraph load_edge_labeled_graph(const char *filename) {
    NodeEdgeLabeledGraph g;

    ifstream file(filename);

    if (file.is_open()) {
        string line;
        getline(file, line);
        unsigned int idx = 0;
        while (getline(file, line)) {
            vector<string> strs;
            boost::split(strs,line,boost::is_any_of(","));
            unsigned int source = stoi(strs[0]);
            unsigned int target = stoi(strs[1]);
            string label = strs[2].substr(3, strs[2].length()-6);
            NodeEdgeLabeledGraph::edge_descriptor edge = boost::add_edge(source, target, {idx++, label}, g).first;
        }

        file.close();
    }
    else {
        cerr << "file is closed" << endl;
    }

    return g;
}

NodeEdgeLabeledGraph load_node_edge_labeled_graph(const char *edges_file, const char *node_labels_file) {
    NodeEdgeLabeledGraph g;

    ifstream n_file(node_labels_file);

    if (n_file.is_open()) {
        string line;
        getline(n_file, line);
        while (getline(n_file, line)) {
            vector<string> strs;
            boost::split(strs,line,boost::is_any_of(","));
            unsigned int source = stoi(strs[0]);
            //unsigned int target = stoi(strs[1]);
            string label = strs[1].substr(4, strs[1].length()-8);
            Node n = boost::add_vertex({source, label}, g);
        }

        n_file.close();
    }
    else {
        cerr << "file is closed" << endl;
    }

    ifstream e_file(edges_file);

    if (e_file.is_open()) {
        string line;
        getline(e_file, line);
        unsigned int idx = 0;
        while (getline(e_file, line)) {
            vector<string> strs;
            boost::split(strs,line,boost::is_any_of(","));
            unsigned int source = stoi(strs[0]);
            unsigned int target = stoi(strs[1]);
            string label = strs[2].substr(3, strs[2].length()-6);
            NodeEdgeLabeledGraph::edge_descriptor edge = boost::add_edge(source, target, {idx++, label}, g).first;
        }

        e_file.close();
    }
    else {
        cerr << "file is closed" << endl;
    }

    return g;
}

std::unordered_map<std::string,int> get_edge_label_distribution(NodeEdgeLabeledGraph &g) {
    NodeEdgeLabeledGraph::vertex_iterator v, vend;
    typename graph_traits < NodeEdgeLabeledGraph >::out_edge_iterator ei, ei_end;

    boost::property_map< NodeEdgeLabeledGraph, edge_name_t >::type label_map = boost::get(boost::edge_name, g);
    std::unordered_map<string,int> label_count_map;

    for (boost::tie(v, vend) = vertices(g); v != vend; ++v) {
        for (boost::tie(ei, ei_end) = boost::out_edges(*v, g); ei != ei_end; ++ei) {
            if(!label_count_map.contains(label_map[*ei]))
                label_count_map.insert(make_pair(label_map[*ei],1));
            else
                label_count_map[label_map[*ei]] = label_count_map[label_map[*ei]] + 1;
        }
    }
    return label_count_map;
}

std::vector<std::string> get_edge_labels(NodeEdgeLabeledGraph &g) {
    NodeEdgeLabeledGraph::vertex_iterator v, vend;
    typename graph_traits < NodeEdgeLabeledGraph >::out_edge_iterator ei, ei_end;

    boost::property_map< NodeEdgeLabeledGraph, edge_name_t >::type label_map = boost::get(boost::edge_name, g);
    std::unordered_set<string> label_count_set;

    for (boost::tie(v, vend) = vertices(g); v != vend; ++v) {
        for (boost::tie(ei, ei_end) = boost::out_edges(*v, g); ei != ei_end; ++ei) {
            label_count_set.insert(label_map[*ei]);
        }
    }
    std::vector<string> labels;
    labels.reserve(label_count_set.size());
    for (auto it = label_count_set.begin(); it != label_count_set.end(); ) {
        labels.push_back(std::move(label_count_set.extract(it++).value()));
    }
    return labels;
}