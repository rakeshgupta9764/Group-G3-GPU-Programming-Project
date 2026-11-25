#ifndef SSSP_H
#define SSSP_H

#include "graph.h"
#include <vector>

using namespace std;

const int SSSP_INF = 1000000000;

// CPU Bellman-Ford reference
vector<int> sssp_cpu_bellman_ford(const Graph& graph, int source, int& iterations);

// GPU execution timing
struct SSSPTiming
{
    float kernel_ms = 0.0f;
    double total_ms = 0.0;
    int iterations = 0;
};

// GPU vertex centric Push using CSR
vector<int> sssp_gpu_push(const Graph& graph, int source,SSSPTiming& timing);

// GPU edge centric using COO
vector<int> sssp_gpu_edge(const Graph& graph,int source,SSSPTiming& timing);


#endif
