#ifndef PATHENUMERATION_ENUMERATOR_HPP
#define PATHENUMERATION_ENUMERATOR_HPP

#include "model.hpp"
#include "trie.hpp"

class PathEnumeratorLabel {
public:
    size_t node_id;
    unsigned int hop_count;
    explicit PathEnumeratorLabel(size_t node_id);
    size_t previous_id;
    PathEnumeratorLabel* previous_pel;
    ~PathEnumeratorLabel();
};

enum Semantics { HOMOMORPHISM, CYPHERMORPHISM, ISOMORPHISM };

class PathEnumerator {
    vector<unsigned long long> cardinalities_vector_el(NodeEdgeLabeledGraph &g, PathPattern &pattern, Semantics sem, int timeout);
    vector<unsigned long long> cardinalities_vector_nel(NodeEdgeLabeledGraph &g, PathPattern &pattern, Semantics sem, int timeout);
public:
    vector<unsigned long long> cardinalities_vector_el_hm(NodeEdgeLabeledGraph &g, PathPattern &pattern);
    vector<unsigned long long> cardinalities_vector_el_cm(NodeEdgeLabeledGraph &g, PathPattern &pattern);
    vector<unsigned long long> cardinalities_vector_el_im(NodeEdgeLabeledGraph &g, PathPattern &pattern);
    vector<unsigned long long> cardinalities_vector_nel_hm(NodeEdgeLabeledGraph &g, PathPattern &pattern);
    vector<unsigned long long> cardinalities_vector_nel_cm(NodeEdgeLabeledGraph &g, PathPattern &pattern);
    vector<unsigned long long> cardinalities_vector_nel_im(NodeEdgeLabeledGraph &g, PathPattern &pattern);

};

#endif
