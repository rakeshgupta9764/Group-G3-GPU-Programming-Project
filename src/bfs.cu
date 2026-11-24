#include "bfs.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <chrono>
#include <algorithm>

using namespace std;

// Check CUDA API errors.
void check_cuda(cudaError_t error, const char* operation)
{

    if (error != cudaSuccess)
    {

        throw runtime_error(string(operation) + ": " + cudaGetErrorString(error) );
    }
}


// --------------------------------------------------
// GPU KERNEL VERTEX-CENTRIC PUSH BFS
// --------------------------------------------------

__global__ void bfs_vertex_push_kernel(const int* row_ptr, const int* col_idx, const int* current_frontier,int* next_frontier,
                                       int* distance, int* next_frontier_count, int num_vertices, int current_level)
{


    int u = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads outside the graph.
    if (u >= num_vertices)
        return;

    // Process only vertices belonging to the current frontier.
    if (current_frontier[u] == 0)
        return;

    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int e = start; e < end; e++)
    {

        int v = col_idx[e];

        // Attempt to discover vertex v.
        int old_distance = atomicCAS(&distance[v], -1, current_level + 1);

        // Only the thread that discovers v for the
        // first time will enter this condition.
        if (old_distance == -1)
        {

            next_frontier[v] = 1;

            atomicAdd(next_frontier_count, 1);
        }
    }
}


// --------------------------------------------------
// GPU VERTEX-CENTRIC PUSH BFS
// --------------------------------------------------

vector<int> bfs_gpu_push(const Graph& graph, int source, BFSTiming& timing)
{

    int V = graph.num_vertices;
    int E = graph.num_edges;

    if (source < 0 || source >= V)
    {
        throw invalid_argument("Invalid BFS source");
    }

    // Host-side initialization.
    vector<int> distance(V, -1);

    vector<int> frontier_current(V, 0);
    vector<int> frontier_next(V, 0);

    distance[source] = 0;
    frontier_current[source] = 1;

    // GPU pointers.
    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    int *d_distance = nullptr;

    int *d_frontier_current = nullptr;
    int *d_frontier_next = nullptr;
    int *d_next_count = nullptr;

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    bool events_created = false;

    timing.kernel_ms = 0.0f;
    timing.total_ms = 0.0;
    timing.iterations = 0;

    auto total_start = chrono::steady_clock::now();

    try {

        // Allocate GPU memory.

        check_cuda(cudaMalloc(&d_row_ptr, (V + 1) * sizeof(int)), "Allocate row_ptr");

        if (E > 0)
        {

            check_cuda(cudaMalloc(&d_col_idx, E * sizeof(int)), "Allocate col_idx");
        }

        check_cuda(cudaMalloc(&d_distance, V * sizeof(int)),"Allocate distance");

        check_cuda(cudaMalloc(&d_frontier_current, V * sizeof(int)),"Allocate current frontier");

        check_cuda(cudaMalloc(&d_frontier_next, V * sizeof(int)), "Allocate next frontier");

        check_cuda(cudaMalloc(&d_next_count, sizeof(int)), "Allocate frontier counter");



        // Copy graph and initial data to GPU.


        check_cuda(cudaMemcpy(d_row_ptr, graph.csr.row_ptr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice), "Copy row_ptr");

        if (E > 0)
	{

            check_cuda(cudaMemcpy(d_col_idx, graph.csr.col_idx.data(), E * sizeof(int), cudaMemcpyHostToDevice), "Copy col_idx");
        }

        check_cuda(cudaMemcpy(d_distance, distance.data(), V * sizeof(int), cudaMemcpyHostToDevice), "Copy distance");

        check_cuda(cudaMemcpy(d_frontier_current, frontier_current.data(), V * sizeof(int), cudaMemcpyHostToDevice), "Copy current frontier");

        check_cuda( cudaMemset(d_frontier_next, 0, V * sizeof(int)), "Initialize next frontier");



        // Create CUDA events for kernel timing.

        check_cuda(cudaEventCreate(&start_event), "Create start event");

        check_cuda(cudaEventCreate(&stop_event), "Create stop event");

        events_created = true;


        // BFS level-by-level execution.

        const int BLOCK_SIZE = 256;

        int grid_size = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;

        int current_level = 0;

        for (int iteration = 0; iteration < V; iteration++)
	{

            // Clear next frontier and counter.
            check_cuda(cudaMemset(d_frontier_next, 0, V * sizeof(int)), "Clear next frontier");

            check_cuda(cudaMemset(d_next_count, 0, sizeof(int)), "Clear frontier counter");

            // Start measuring this kernel.
            check_cuda(cudaEventRecord(start_event), "Record start event");

            bfs_vertex_push_kernel<<<grid_size, BLOCK_SIZE>>>(d_row_ptr, d_col_idx, d_frontier_current, d_frontier_next,
                                                              d_distance, d_next_count, V, current_level);

            check_cuda(cudaGetLastError(), "Launch BFS push kernel");

            check_cuda(cudaEventRecord(stop_event), "Record stop event");

            check_cuda(cudaEventSynchronize(stop_event), "Synchronize BFS kernel");

            float iteration_time = 0.0f;

            check_cuda(cudaEventElapsedTime(&iteration_time, start_event, stop_event), "Calculate kernel time");

            timing.kernel_ms += iteration_time;


            // Copy newly discovered vertex count to CPU.
            int next_count = 0;

            check_cuda(cudaMemcpy(&next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost), "Copy frontier count");

            timing.iterations++;

            // No newly discovered vertices: BFS is complete.
            if (next_count == 0)
                break;

            // Swap current and next frontier arrays.
            swap(d_frontier_current, d_frontier_next);

            current_level++;
        }


        // Copy final distances back to CPU.


        check_cuda(cudaMemcpy(distance.data(), d_distance, V * sizeof(int), cudaMemcpyDeviceToHost), "Copy final distances");



        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        events_created = false;

        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_distance);

        cudaFree(d_frontier_current);
        cudaFree(d_frontier_next);
        cudaFree(d_next_count);

    }

    catch (...)
    {

        if (events_created)
        {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }

        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_distance);

        cudaFree(d_frontier_current);
        cudaFree(d_frontier_next);
        cudaFree(d_next_count);

        throw;
    }

    auto total_end = chrono::steady_clock::now();

    timing.total_ms = chrono::duration<double, milli>(total_end - total_start).count();

    return distance;
}


// --------------------------------------------------
// GPU KERNEL VERTEX-CENTRIC PULL BFS
// --------------------------------------------------

__global__ void bfs_vertex_pull_kernel(const int* col_ptr, const int* row_idx, const int* current_frontier, int* next_frontier,
                                       int* distance, int* next_frontier_count, int num_vertices, int current_level )
{

    // Each thread is assigned one destination vertex.
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= num_vertices)
        return;

    // Already visited vertices do not need processing.
    if (distance[v] != -1)
        return;

    // Inspect incoming neighbours of v using CSC.
    int start = col_ptr[v];
    int end = col_ptr[v + 1];

    for (int e = start; e < end; e++)
    {

        int u = row_idx[e];

        // Check whether an incoming neighbour belongs to the current BFS frontier.
        if (current_frontier[u] == 1)
        {

            distance[v] = current_level + 1;

            next_frontier[v] = 1;

            atomicAdd(next_frontier_count, 1);

            //One incoming frontier is enough.
            break;
        }
    }
}

// --------------------------------------------------
// HOST FUNCTION GPU VERTEX-CENTRIC PULL BFS
// --------------------------------------------------

vector<int> bfs_gpu_pull(const Graph& graph, int source, BFSTiming& timing)
{

    int V = graph.num_vertices;
    int E = graph.num_edges;

    if (source < 0 || source >= V)
    {
        throw invalid_argument("Invalid BFS source");
    }

    vector<int> distance(V, -1);
    vector<int> frontier_current(V, 0);
    vector<int> frontier_next(V, 0);

    distance[source] = 0;
    frontier_current[source] = 1;

    int *d_col_ptr = nullptr;
    int *d_row_idx = nullptr;
    int *d_distance = nullptr;

    int *d_frontier_current = nullptr;
    int *d_frontier_next = nullptr;
    int *d_next_count = nullptr;

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    bool events_created = false;

    timing.kernel_ms = 0.0f;
    timing.total_ms = 0.0;
    timing.iterations = 0;

    auto total_start = chrono::steady_clock::now();

    try {

        // Allocate GPU memory.
        check_cuda(cudaMalloc(&d_col_ptr, (V + 1) * sizeof(int)), "Allocate col_ptr");

        if (E > 0)
	{
            check_cuda(cudaMalloc(&d_row_idx, E * sizeof(int)), "Allocate row_idx");
        }

        check_cuda(cudaMalloc(&d_distance, V * sizeof(int)), "Allocate distance");

        check_cuda(cudaMalloc(&d_frontier_current, V * sizeof(int)), "Allocate current frontier");

        check_cuda(cudaMalloc(&d_frontier_next, V * sizeof(int)),"Allocate next frontier");

        check_cuda(cudaMalloc(&d_next_count, sizeof(int)), "Allocate frontier counter");


        // Copy CSC and initial arrays to GPU.
        check_cuda(cudaMemcpy(d_col_ptr, graph.csc.col_ptr.data(),(V + 1) * sizeof(int), cudaMemcpyHostToDevice), "Copy col_ptr");

        if (E > 0)
	{
            check_cuda(cudaMemcpy(d_row_idx, graph.csc.row_idx.data(), E * sizeof(int), cudaMemcpyHostToDevice), "Copy row_idx");
        }

        check_cuda(cudaMemcpy(d_distance, distance.data(), V * sizeof(int), cudaMemcpyHostToDevice), "Copy distance");

        check_cuda(cudaMemcpy(d_frontier_current, frontier_current.data(), V * sizeof(int), cudaMemcpyHostToDevice), "Copy current frontier");

        check_cuda(cudaMemset(d_frontier_next, 0, V * sizeof(int)), "Initialize next frontier");


        // Create CUDA events.
        check_cuda(cudaEventCreate(&start_event),"Create start event");

        check_cuda(cudaEventCreate(&stop_event), "Create stop event");

        events_created = true;


        // BFS execution.
        const int BLOCK_SIZE = 256;

        int grid_size = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;

        int current_level = 0;

        for (int iteration = 0; iteration < V; iteration++)
	{

            check_cuda(cudaMemset(d_frontier_next, 0, V * sizeof(int)), "Clear next frontier");

            check_cuda(cudaMemset(d_next_count, 0, sizeof(int)), "Clear frontier counter");

            check_cuda(cudaEventRecord(start_event), "Record start event");

            bfs_vertex_pull_kernel<<<grid_size, BLOCK_SIZE>>>(d_col_ptr, d_row_idx, d_frontier_current, d_frontier_next,
                                      d_distance, d_next_count, V, current_level);

            check_cuda(cudaGetLastError(), "Launch BFS pull kernel");

            check_cuda(cudaEventRecord(stop_event),"Record stop event");

            check_cuda(cudaEventSynchronize(stop_event), "Synchronize BFS kernel");

            float iteration_time = 0.0f;

            check_cuda(cudaEventElapsedTime(&iteration_time, start_event, stop_event), "Calculate kernel time");

            timing.kernel_ms += iteration_time;

            int next_count = 0;

            check_cuda(cudaMemcpy(&next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost), "Copy frontier count");

            timing.iterations++;

            if (next_count == 0)
                break;

            swap(d_frontier_current, d_frontier_next);

            current_level++;
        }


        // Copy final distances to CPU.
        check_cuda(cudaMemcpy(distance.data(), d_distance, V * sizeof(int), cudaMemcpyDeviceToHost), "Copy final distances");


        // Free resources.
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        events_created = false;

        cudaFree(d_col_ptr);
        cudaFree(d_row_idx);
        cudaFree(d_distance);

        cudaFree(d_frontier_current);
        cudaFree(d_frontier_next);
        cudaFree(d_next_count);

    }

    catch (...)
    {

        if (events_created)
        {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }

        cudaFree(d_col_ptr);
        cudaFree(d_row_idx);
        cudaFree(d_distance);

        cudaFree(d_frontier_current);
        cudaFree(d_frontier_next);
        cudaFree(d_next_count);

        throw;
    }

    auto total_end = chrono::steady_clock::now();

    timing.total_ms = chrono::duration<double, std::milli>(total_end - total_start).count();

    return distance;
}


// --------------------------------------------------
// GPU KERNEL EDGE-CENTRIC BFS
// --------------------------------------------------

__global__ void bfs_edge_kernel(const int* src, const int* dst, int* distance, const int* current_frontier,
                                int* next_frontier, int* next_frontier_count, int num_edges, int current_level)
{


    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= num_edges)
        return;

    // Get source and destination of this edge.
    int u = src[e];
    int v = dst[e];

    // Process only edges whose source belongs
    // to the current BFS frontier.
    if (current_frontier[u] == 0)
        return;

    int old_distance = atomicCAS(&distance[v], -1, current_level + 1);

    // Only the thread that discovers v succeeds.
    if (old_distance == -1)
    {

        next_frontier[v] = 1;

        atomicAdd(next_frontier_count, 1);
    }
}


// --------------------------------------------------
// HOST FUNCTION GPU EDGE-CENTRIC BFS
// --------------------------------------------------

vector<int> bfs_gpu_edge(const Graph& graph,int source, BFSTiming& timing)
{

    int V = graph.num_vertices;
    int E = graph.num_edges;

    if (source < 0 || source >= V)
    {
        throw invalid_argument("Invalid BFS source");
    }

    vector<int> distance(V, -1);
    vector<int> frontier_current(V, 0);

    distance[source] = 0;
    frontier_current[source] = 1;

    int *d_src = nullptr;
    int *d_dst = nullptr;
    int *d_distance = nullptr;

    int *d_frontier_current = nullptr;
    int *d_frontier_next = nullptr;
    int *d_next_count = nullptr;

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    bool events_created = false;

    timing.kernel_ms = 0.0f;
    timing.total_ms = 0.0;
    timing.iterations = 0;

    auto total_start = chrono::steady_clock::now();

    try
    {

        // Allocate GPU memory.

        if (E > 0)
  	{

            check_cuda(cudaMalloc(&d_src, E * sizeof(int)), "Allocate COO source");

            check_cuda(cudaMalloc(&d_dst, E * sizeof(int)), "Allocate COO destination");
        }

        check_cuda(cudaMalloc(&d_distance, V * sizeof(int)),"Allocate distance");

        check_cuda(cudaMalloc(&d_frontier_current, V * sizeof(int)),"Allocate current frontier");

        check_cuda(cudaMalloc(&d_frontier_next, V * sizeof(int)),"Allocate next frontier");

        check_cuda(cudaMalloc(&d_next_count, sizeof(int)),"Allocate frontier counter");


        //Copy COO and initial data to GPU.

        if (E > 0)
	{

            check_cuda(cudaMemcpy(d_src,graph.coo.src.data(),E * sizeof(int),cudaMemcpyHostToDevice), "Copy COO source");

            check_cuda(cudaMemcpy(d_dst,graph.coo.dst.data(),E * sizeof(int),cudaMemcpyHostToDevice), "Copy COO destination");
        }

        check_cuda(cudaMemcpy(d_distance,distance.data(),V * sizeof(int),cudaMemcpyHostToDevice),"Copy distance");

        check_cuda(cudaMemcpy(d_frontier_current,frontier_current.data(),V * sizeof(int),cudaMemcpyHostToDevice),"Copy current frontier");


        //Create CUDA events.

        check_cuda(cudaEventCreate(&start_event), "Create start event");

        check_cuda(cudaEventCreate(&stop_event),"Create stop event");

        events_created = true;


        // BFS execution.

        const int BLOCK_SIZE = 256;

        int grid_size = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Handle graphs with zero edges.
        if (grid_size == 0)
            grid_size = 1;

        int current_level = 0;

        for (int iteration = 0; iteration < V; iteration++)
	{

            check_cuda(cudaMemset(d_frontier_next, 0, V * sizeof(int)),"Clear next frontier");

            check_cuda(cudaMemset(d_next_count, 0, sizeof(int)),"Clear frontier counter");

            check_cuda(cudaEventRecord(start_event),"Record start event");

            bfs_edge_kernel<<<grid_size, BLOCK_SIZE>>>(d_src, d_dst, d_distance, d_frontier_current,
                                                       d_frontier_next, d_next_count,E, current_level);

            check_cuda(cudaGetLastError(),"Launch edge-centric BFS kernel");

            check_cuda(cudaEventRecord(stop_event),"Record stop event");

            check_cuda(cudaEventSynchronize(stop_event), "Synchronize edge-centric BFS");

            float iteration_time = 0.0f;

            check_cuda(cudaEventElapsedTime(&iteration_time,start_event,stop_event), "Calculate kernel time");

            timing.kernel_ms += iteration_time;

            int next_count = 0;

            check_cuda(cudaMemcpy(&next_count,d_next_count,sizeof(int),cudaMemcpyDeviceToHost),"Copy frontier count");

            timing.iterations++;

            if (next_count == 0)
                break;

            swap(d_frontier_current, d_frontier_next);

            current_level++;
        }


        // Copy results back to CPU.

        check_cuda(cudaMemcpy(distance.data(), d_distance,V * sizeof(int),cudaMemcpyDeviceToHost), "Copy final distances");


        //Free GPU memory.

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        events_created = false;

        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_distance);

        cudaFree(d_frontier_current);
        cudaFree(d_frontier_next);
        cudaFree(d_next_count);

    }

    catch (...)
    {

        if (events_created)
	{
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }

        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_distance);

        cudaFree(d_frontier_current);
        cudaFree(d_frontier_next);
        cudaFree(d_next_count);

        throw;
    }

    auto total_end = chrono::steady_clock::now();

    timing.total_ms = chrono::duration<double, std::milli>(total_end - total_start).count();

    return distance;
}
