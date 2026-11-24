#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>

// Represents one directed edge
struct Edge {
    int src;
    int dst;
    int weight;
};

// Compressed Sparse Row representation.
struct CSRGraph {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<int> weights;
};

// Compressed Sparse Column representation.
struct CSCGraph {
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
    std::vector<int> weights;
};

// Coordinate representation.
struct COOGraph {
    std::vector<int> src;
    std::vector<int> dst;
    std::vector<int> weights;
};

// Complete graph containing all three representations
struct Graph {
    int num_vertices = 0;
    int num_edges = 0;

    std::vector<Edge> edges;

    std::vector<int> out_degree;

    CSRGraph csr;
    CSCGraph csc;
    COOGraph coo;
};

// Read a graph from an edgelist file.
bool load_graph(const std::string& filename, Graph& graph);

// Construct CSR, CSC and COO from the edge list.
void build_graph_formats(Graph& graph);

// Display basic graph information
void print_graph_summary(const Graph& graph);

#endif
