#ifndef BFS_H
#define BFS_H

#include "graph.h"
#include <vector>

using namespace std;

// CPU BFS reference.
vector<int> bfs_cpu(const Graph& graph, int source);

// Timing information for GPU BFS.
struct BFSTiming
{
    float kernel_ms = 0.0f;
    double total_ms = 0.0;
    int iterations = 0;
};

// GPU vertex-centric push BFS.
vector<int> bfs_gpu_push(const Graph& graph, int source,BFSTiming& timing);

// GPU vertex-centric pull BFS.
vector<int> bfs_gpu_pull(const Graph& graph, int source,BFSTiming& timing);

// GPU edge-centric BFS.
vector<int> bfs_gpu_edge(const Graph& graph,int source, BFSTiming& timing);

#endif
