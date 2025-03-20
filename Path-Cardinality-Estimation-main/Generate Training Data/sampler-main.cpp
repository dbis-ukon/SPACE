#include "sampler.hpp"
#include "trie.hpp"

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <chrono>

using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;

void print_help() {
    cerr << "Wrong arguments. Please run the executable as follows:" << endl;
    cerr << "./sampler -f [GRAPH_FILE] -o [OUTPUT_FILE] <options>" << endl;
    cerr << endl;
    cerr << "Options include" << endl;
    cerr << "-r [INTEGER]\tMaximum number of random walks to be executed. Sampler finishes if this number is reached." << endl;
    cerr << "-s [INTEGER]\tMaximum size of pattern in terms of edges. Sampler generates patterns of size between 2 and the defined number. " << endl;
    cerr << "-p [edges|all]\tThe type of pattern that is sampled. A pattern contains either edge labels only, or both edge and node labels." << endl;
    cerr << "-n [FILE_NAME]\tInput file containing node labels. If -p is set to edges, then this parameter is ignored." << endl;
}

bool validate_input(const string& graph_file,
                    const string& node_labels_file,
                    int max_random_walks,
                    int max_pattern_size,
                    const string& output_file,
                    const string& pattern_type) {

    if (graph_file.empty() || output_file.empty() || pattern_type.empty()) {
        cout << "Empty filename" << endl;
        return false;
    }

    if (pattern_type != "all" && pattern_type != "edges") {
        cout << "Pattern type" << endl;
        return false;
    }

    if (max_pattern_size <= 0 || max_random_walks <= 0) {
        cout << "Numeric" << endl;
        return false;
    }

    if(pattern_type == "all" && node_labels_file.empty()) {
        cout << "Other" << endl;
        return false;
    }

    return true;
}

int main(int argc, char **argv) {

    string edges_file;
    string node_labels_file;
    int max_random_walks = 0;
    int max_pattern_size = 0;
    string output_file;
    string pattern_type;

    int opt;
    while ((opt = getopt(argc, argv, "f:r:s:o:p:n:")) != -1) {
        switch(opt) {
            case 'f':
                edges_file = string(optarg);
                break;
            case 'n':
                node_labels_file = string(optarg);
                break;
            case 'r':
                max_random_walks = stoi(string(optarg));
                break;
            case 's':
                max_pattern_size = stoi(string(optarg));
                break;
            case 'o':
                output_file = string(optarg);
                break;
            case 'p':
                pattern_type = string(optarg);
                break;
        }
    }

    bool valid_input = validate_input(edges_file, node_labels_file, max_random_walks, max_pattern_size, output_file, pattern_type);

    if(!valid_input) {
        print_help();
        return 1;
    }

    if(max_pattern_size <= 0 || max_random_walks <= 0 || strcmp(output_file.c_str(), "") == 0) {
        std::cerr << "Wrong arguments" << endl;
        exit(1);
    }

    std::cerr << "Input edge labeled graph = " << edges_file << endl;
    std::cerr << "Max pattern size = " << max_pattern_size << endl;
    std::cerr << "Number of patterns = " << max_random_walks << endl;
    std::cerr << "Output file = " << output_file << endl;

    // Loading graph
    auto p1 = high_resolution_clock::now();
    NodeEdgeLabeledGraph g;
    if(strcmp(node_labels_file.c_str(), "") == 0) {
        g = load_edge_labeled_graph(edges_file.c_str());
    }
    else {
        g = load_node_edge_labeled_graph(edges_file.c_str(), node_labels_file.c_str());
    }
    auto p2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double_pre = p2 - p1;

    std::cerr << "Number of nodes: " << boost::num_vertices(g) << endl;
    std::cerr << "Number of edges: " << boost::num_edges(g) << endl;
    std::cerr << "Loading finished in " << ms_double_pre.count() << "ms." << endl;

    std::vector<string> edge_labels = get_edge_labels(g);

    for(string s : edge_labels)
        cout << " - " << s << endl;

    cout << edge_labels.size() << endl;

    exit(1);

    vector<PathPattern> sampled_patterns;

    PatternSampler sampler;
    p1 = high_resolution_clock::now();
    if(pattern_type == "edges")
        sampled_patterns = sampler.random_walk_sampler(g, max_random_walks, max_pattern_size);
    else
        sampled_patterns = sampler.random_walk_sampler_2(g, max_random_walks, max_pattern_size);
    p2 = high_resolution_clock::now();
    ms_double_pre = p2 - p1;
    std::cerr << "Sampling finished in " << ms_double_pre.count() << "ms." << endl;
    std::cerr << "Number of sampled patterns = " << sampled_patterns.size() << endl;
    ofstream out_file(output_file);
    for(auto pattern : sampled_patterns) {
        out_file << pattern.at(0);
        for(unsigned int i=1;i<pattern.size();i++)
            out_file << " " << pattern.at(i);
        out_file << endl;
    }
    out_file.close();
    std::cerr << "Writing patterns to file completed." << endl;

    return 0;
}
