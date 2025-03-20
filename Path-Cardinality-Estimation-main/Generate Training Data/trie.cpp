#include "trie.hpp"

using namespace std;

EdgeLabelTrieWithCardinalities::EdgeLabelTrieWithCardinalities() {
    this->root = new EdgeLabelTrieWithCardinalitiesEntry();
    this->root->label = "_ROOT";
    this->height = 0;
};

long long EdgeLabelTrieWithCardinalities::cardinality(vector<string> &label_seq) {
    if(label_seq.size() > this->height)
        return -1;
    EdgeLabelTrieWithCardinalitiesEntry *current_entry = this->root;
    return this->recursive_cardinality_retrieval(current_entry, 0, label_seq);
}

long long EdgeLabelTrieWithCardinalities::recursive_cardinality_retrieval(EdgeLabelTrieWithCardinalitiesEntry* current_entry, int idx, vector<string> &label_seq) {
    int cardinality = -1;
    if(idx < label_seq.size()) {
        string wanted_label = label_seq[idx];
        cardinality = recursive_cardinality_retrieval(current_entry->children[wanted_label], idx+1, label_seq);
    }
    else {
        if(current_entry->children.size() == 0)
            cardinality = current_entry->cardinality;
    }
    return cardinality;
}

int EdgeLabelTrieWithCardinalities::increment_cardinality(vector<string> &path_label_sequence) {
    EdgeLabelTrieWithCardinalitiesEntry *current_entry = this->root;
    return this->recursive_increment(current_entry, 0, path_label_sequence);
}

int EdgeLabelTrieWithCardinalities::recursive_increment(EdgeLabelTrieWithCardinalitiesEntry* current_entry, int idx, vector<string> &label_seq) {
    if(idx < label_seq.size()) {
        string wanted_label = label_seq[idx];
        if(!current_entry->children.contains(wanted_label)) {
            return -1;
        }
        return this->recursive_increment(current_entry->children[wanted_label], idx+1, label_seq);
    }
    else {
        current_entry->cardinality = current_entry->cardinality + 1;
        return (int)current_entry->cardinality;
    }
}

void EdgeLabelTrieWithCardinalities::initialize(vector<PathPattern> &patterns) {
    for(auto path_label_sequence : patterns) {
        if (this->height < path_label_sequence.size())
            this->height = path_label_sequence.size();
        EdgeLabelTrieWithCardinalitiesEntry *current_entry = this->root;
        this->recursive_update(current_entry, 0, path_label_sequence);
    }
}

void EdgeLabelTrieWithCardinalities::recursive_update(EdgeLabelTrieWithCardinalitiesEntry* current_entry, int idx, vector<string> &label_seq) {
    if(idx < label_seq.size()) {
        string wanted_label = label_seq[idx];
        if(!current_entry->children.contains(wanted_label)) {
            EdgeLabelTrieWithCardinalitiesEntry* new_label = new EdgeLabelTrieWithCardinalitiesEntry();
            new_label->label = wanted_label;
            new_label->cardinality = 0;
            current_entry->children.insert(make_pair(wanted_label,new_label));
        }
        recursive_update(current_entry->children[wanted_label], idx+1, label_seq);
    }
}

/*
 * PRINT Section
 *
 * The following functions are only useful to print the trie.
 */

void EdgeLabelTrieWithCardinalities::print_non_zero_cardinalities() {
    EdgeLabelTrieWithCardinalitiesEntry *current_entry = this->root;
    std::vector<string> path_label_sequence;
    this->recursive_print(current_entry, 0, path_label_sequence, 1,false);
}

void EdgeLabelTrieWithCardinalities::print_cardinalities() {
    EdgeLabelTrieWithCardinalitiesEntry *current_entry = this->root;
    std::vector<string> path_label_sequence;
    this->recursive_print(current_entry, 0, path_label_sequence, 0, false);
}

void EdgeLabelTrieWithCardinalities::print_leaf_cardinalities() {
    EdgeLabelTrieWithCardinalitiesEntry *current_entry = this->root;
    PathPattern pattern;
    this->recursive_print(current_entry, 0, pattern, 0, true);
}

void EdgeLabelTrieWithCardinalities::recursive_print(EdgeLabelTrieWithCardinalitiesEntry* current_entry, unsigned int level, PathPattern &pattern, unsigned int size_limit, bool only_leaves) {
    if(current_entry->label != "_ROOT" && current_entry->cardinality >= size_limit) {
        if(!only_leaves || (only_leaves && current_entry->children.size() == 0)) {
            for(unsigned int i=0;i<pattern.size();i++) {
                cout << pattern[i] << " ";
            }
            cout << ": " << current_entry->cardinality << endl;
        }
    }
    if(level >= height)
        return;

    for(auto it : current_entry->children) {
        PathPattern label_seq_next = pattern;
        label_seq_next.push_back(it.first);
        this->recursive_print(it.second, level+1, label_seq_next, size_limit, only_leaves);
        label_seq_next.clear();
    }
}


