#include "enumerator.hpp"
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
    cerr << "./enumerator -f [GRAPH_FILE] -q [PATTERNS_FILE] -o [OUTPUT_FILE] <options>" << endl;
    cerr << endl;
    cerr << "Options include" << endl;
    cerr << "-s [hm|cm|im]\tThe semantics used for subgraph matching; hm for homomorphism, cm for cyphermorphism and im for isomorphism." << endl;
    cerr << "-p [edges|all]\tThe type of pattern that is given as input. A pattern contains either edge labels only, or both edge and node labels." << endl;
    cerr << "-n [FILE_NAME]\tInput file containing node labels. If -p is set to edges, then this parameter is ignored." << endl;
}

void print_cypher(PathPattern query, const string& pattern_type) {
    int nodeCount = 0;
    int relCount = 0;
    if(pattern_type == "edges") {
        cout << "MATCH (n" << nodeCount++ << ")-[r" << relCount++ << ":" << query.at(0) << "]->(n" << nodeCount
             << ")";
        for (unsigned int i = 1; i < query.size(); i++) {
            cout << ", (n" << nodeCount++ << ")-[r" << relCount++ << ":" << query.at(i) << "]->(n" << nodeCount
                 << ")";
        }
        cout << " RETURN n0";

        for (int i = 0; i < relCount; i++)
            cout << ", r" << i << ", n" << (i + 1);
        cout << endl;
    }
    else {
        cout << "MATCH (n" << nodeCount++ << ":"<< query.at(0) << ")-[r"<<relCount++ << ":" << query.at(1) << "]->(n" << nodeCount << ":" << query.at(2) << ")";
        for(unsigned int i=2;i<query.size()-2;i+=2) {
            cout << ", (n" << nodeCount++ << ":"<< query.at(i) << ")-[r"<<relCount++ << ":" << query.at(i+1) << "]->(n" << nodeCount << ":" << query.at(i+2) << ")";
        }
        cout << " RETURN n0";

        for(int i=0;i<relCount;i++)
            cout << ", r" << i << ", n" << (i+1);
        cout << endl;
    }
}

bool validate_input(const string& graph_file,
                    const string& query_file,
                    const string& output_file,
                    const string& semantics,
                    const string& pattern_type,
                    const string& node_labels_file) {

    if(graph_file.empty() || query_file.empty() || output_file.empty() || semantics.empty() || pattern_type.empty())
        return false;

    if(pattern_type != "all" || pattern_type != "edges")
        return false;

    if(pattern_type != "all" || pattern_type != "edges")
        return false;

    if(pattern_type == "all" && node_labels_file.empty())
        return false;

    return true;
}

int main(int argc, char **argv) {

    string edges_file;
    string query_file;
    string output_file;
    string semantics;
    string pattern_type;
    string node_labels_file;

    int opt;
    while ((opt = getopt(argc, argv, "f:q:o:s:p:n:")) != -1) {
        switch(opt) {
            case 'f':
                edges_file = string(optarg);
                break;
            case 'q':
                query_file = string(optarg);
                break;
            case 'o':
                output_file = string(optarg);
                break;
            case 's':
                semantics = string(optarg);
                break;
            case 'p':
                pattern_type = string(optarg);
                break;
            case 'n':
                node_labels_file = string(optarg);
                break;
        }
    }

    if(!validate_input) {
        print_help();
        return 1;
    }

    std::cerr << "Input edge labeled graph = " << edges_file << endl;
    std::cerr << "Input patterns = " << query_file << endl;
    std::cerr << "Input patterns type  = " << pattern_type << endl;
    std::cerr << "Semantics used for matching  = ";
    if(semantics == "hm")
        std::cerr << "Homomorphism" << endl;
    else if(semantics == "cm")
        std::cerr << "Cyphermorphism" << endl;
    else if(semantics == "im")
        std::cerr << "Isomorphism" << endl;
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
    std::cerr << "Number of nodes = " << boost::num_vertices(g) << endl;
    std::cerr << "Number of edges = " << boost::num_edges(g) << endl;

    // Loading patterns
    vector<PathPattern> sampled_patterns;
    ifstream file(query_file.c_str());
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<string> pattern;
            boost::split(pattern,line,boost::is_any_of(" "));
            sampled_patterns.push_back(pattern);
        }
        file.close();
    }
    else {
        cerr << "Pattern file is closed" << endl;
    }
    std::cerr << "Number of query patterns loaded = " << sampled_patterns.size() << endl;
    std::cerr << "Loading finished in " << ms_double_pre.count() << "ms." << endl;

    PathEnumerator enumerator;

    ofstream out_file(output_file);

    p1 = high_resolution_clock::now();
    for(auto query : sampled_patterns) {
        vector<unsigned long long> v;
        if(pattern_type == "edges") {
            if(semantics == "hm")
                v = enumerator.cardinalities_vector_el_hm(g, query);
            else if(semantics == "cm")
                v = enumerator.cardinalities_vector_el_cm(g, query);
            else if(semantics == "im")
                v = enumerator.cardinalities_vector_el_im(g, query);
        }
        else {
            if(semantics == "hm")
                v = enumerator.cardinalities_vector_nel_hm(g, query);
            else if(semantics == "cm")
                v = enumerator.cardinalities_vector_nel_cm(g, query);
            else if(semantics == "im")
                v = enumerator.cardinalities_vector_nel_im(g, query);
        }
        for(const auto& str : query)
            out_file << str << " ";
        out_file << ": ";
        for(auto card : v)
            out_file << card << " ";
        out_file << endl;

        // Code to export patterns to Cypher queries for Graphflow.
        //print_cypher(query, pattern_type);
    }
    p2 = high_resolution_clock::now();
    ms_double_pre = p2 - p1;
    out_file.close();
    std::cerr << "Vectors computated and writen to file in " << ms_double_pre.count() << "ms." << endl;

    return 0;
}
