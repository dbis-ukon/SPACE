#ifndef PATHENUMERATION_TRIE_HPP
#define PATHENUMERATION_TRIE_HPP

#include "model.hpp"
#include<iostream>
#include<unordered_map>

using namespace std;

class EdgeLabelTrieWithCardinalitiesEntry {
public:
    unsigned int cardinality = 0;
    string label;
    std::unordered_map<string, EdgeLabelTrieWithCardinalitiesEntry*> children;
};

class EdgeLabelTrieWithCardinalities {
private:
    EdgeLabelTrieWithCardinalitiesEntry* root;

    long long recursive_cardinality_retrieval(EdgeLabelTrieWithCardinalitiesEntry* current_entry, int idx, PathPattern &pattern);
    void recursive_print(EdgeLabelTrieWithCardinalitiesEntry* current_entry, unsigned int level, PathPattern &pattern, unsigned int size_limit, bool ony_leaves);
    int recursive_increment(EdgeLabelTrieWithCardinalitiesEntry *pEntry, int i, PathPattern &pattern);
    void recursive_update(EdgeLabelTrieWithCardinalitiesEntry *pEntry, int i, PathPattern &pattern);
public:
    EdgeLabelTrieWithCardinalities();
    unsigned int height;
    long long cardinality(PathPattern &pattern);
    int increment_cardinality(PathPattern &pattern);
    void initialize(vector<PathPattern> &patterns);

    void print_non_zero_cardinalities();
    void print_cardinalities();
    void print_leaf_cardinalities();
};


#endif
