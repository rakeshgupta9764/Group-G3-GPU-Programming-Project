
#ifndef PAGERANK_H
#define PAGERANK_H
#include "graph.h"
#include <vector>

using namespace std;

constexpr float PAGERANK_DAMPING = 0.85f;
constexpr float PAGERANK_TOLERANCE = 0.000001f;
constexpr int PAGERANK_MAX_ITERATIONS = 100;

// CPU PageRank reference
vector<float> pagerank_cpu(const Graph& graph, int& iterations);

// GPU timing information
struct PageRankTiming
{
    float kernel_ms = 0.0f;
    double total_ms = 0.0;
    int iterations = 0;
};

// GPU vertex-centric Push using CSR
vector<float> pagerank_gpu_push( const Graph& graph,PageRankTiming& timing);

// GPU vertex-centric Pull using CSC
vector<float> pagerank_gpu_pull(const Graph& graph,PageRankTiming& timing);

#endif
