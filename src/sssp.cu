#include "sssp.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <string>

using namespace std;

// Check CUDA errors
void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw runtime_error(string(operation) + ": " + cudaGetErrorString(error));
    }
}

// Vertex centric Push Bellman Ford kernel
__global__ void sssp_push_kernel(const int* row_ptr, const int* col_idx, const int* weights,
                                 const int* distance, int* next_distance, int* changed, int V)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V)
        return;

    int du = distance[u];

    if (du == 1000000000)
        return;

    // Process outgoing edges of vertex u
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++)
    {
        int v = col_idx[e];

        long long candidate = (long long)du + weights[e];

        if (candidate <= -1000000000 || candidate >= 1000000000)
            continue;

        // Multiple vertices may update the same destination.
        int old = atomicMin(&next_distance[v], (int)candidate);

        if (candidate < old)
            atomicExch(changed, 1);
    }
}

// Edge centric Bellman Ford kernel
__global__ void sssp_edge_kernel(const int* src, const int* dst, const int* weights,
                                 const int* distance, int* next_distance, int* changed,int E)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= E)
        return;

    int u = src[e];
    int v = dst[e];

    int du = distance[u];

    if (du == 1000000000)
        return;

    long long candidate = (long long)du + weights[e];

    if (candidate <= -1000000000 || candidate >= 1000000000)
        return;

    int old = atomicMin(&next_distance[v], (int)candidate);

    if (candidate < old)
        atomicExch(changed, 1);
}

// Common GPU execution function
vector<int> run_sssp_gpu(const Graph& graph, int source,SSSPTiming& timing,bool edge_centric)
{
    timing = SSSPTiming{};

    int V = graph.num_vertices;
    int E = graph.num_edges;

    if (source < 0 || source >= V)
        throw invalid_argument("Invalid source vertex");

    auto total_start = chrono::high_resolution_clock::now();

    int* d_first = nullptr;
    int* d_second = nullptr;
    int* d_weights = nullptr;

    int* d_distance = nullptr;
    int* d_next_distance = nullptr;
    int* d_changed = nullptr;

    cudaEvent_t start, stop;

    int first_count = edge_centric ? E : V + 1;

    try
    {
        // Allocate graph data
        if (first_count > 0)
            check_cuda(cudaMalloc(&d_first, first_count * sizeof(int)),"Allocate graph array");

        if (E > 0)
        {
            check_cuda(cudaMalloc(&d_second, E * sizeof(int)),"Allocate destination array");

            check_cuda(cudaMalloc(&d_weights, E * sizeof(int)),"Allocate weights");
        }

        // Copy CSR or COO data to GPU
        if (edge_centric)
        {
            if (E > 0)
            {
                check_cuda(cudaMemcpy(d_first, graph.coo.src.data(),E * sizeof(int), cudaMemcpyHostToDevice), "Copy COO source");

                check_cuda(cudaMemcpy(d_second, graph.coo.dst.data(),E * sizeof(int), cudaMemcpyHostToDevice), "Copy COO destination");

                check_cuda(cudaMemcpy(d_weights, graph.coo.weights.data(),E * sizeof(int), cudaMemcpyHostToDevice), "Copy COO weights");
            }
        }
        else
        {
            check_cuda(cudaMemcpy(d_first, graph.csr.row_ptr.data(),(V + 1) * sizeof(int), cudaMemcpyHostToDevice), "Copy CSR row pointers");

            if (E > 0)
            {
                check_cuda(cudaMemcpy(d_second, graph.csr.col_idx.data(),E * sizeof(int), cudaMemcpyHostToDevice),"Copy CSR destinations");

                check_cuda(cudaMemcpy(d_weights, graph.csr.weights.data(),E * sizeof(int), cudaMemcpyHostToDevice), "Copy CSR weights");
            }
        }

        // Allocate distance arrays
        check_cuda(cudaMalloc(&d_distance, V * sizeof(int)), "Allocate distance array");

        check_cuda(cudaMalloc(&d_next_distance, V * sizeof(int)), "Allocate next distance array");

        check_cuda(cudaMalloc(&d_changed, sizeof(int)), "Allocate changed flag");

        // Initialize distances
        vector<int> initial_distance(V, SSSP_INF);

        initial_distance[source] = 0;

        check_cuda(cudaMemcpy(d_distance, initial_distance.data(),V * sizeof(int), cudaMemcpyHostToDevice), "Initialize distances");

        check_cuda(cudaEventCreate(&start), "Create start event");
        check_cuda(cudaEventCreate(&stop), "Create stop event");

        const int BLOCK_SIZE = 256;

        // Bellman-Ford: at most V-1 iterations
        for (int iteration = 0; iteration < V - 1; iteration++)
        {
            // Preserve previous distances
            check_cuda(cudaMemcpy(d_next_distance, d_distance,V * sizeof(int), cudaMemcpyDeviceToDevice), "Copy distances");

            check_cuda(cudaMemset(d_changed, 0, sizeof(int)), "Reset changed flag");

            check_cuda(cudaEventRecord(start), "Record start event");

            if (edge_centric)
            {
                if (E > 0)
                {
                    int blocks = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    sssp_edge_kernel<<<blocks, BLOCK_SIZE>>>(d_first,d_second,d_weights,d_distance,
                                                             d_next_distance,d_changed,E );
                }
            }
            else
            {
                int blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;

                sssp_push_kernel<<<blocks, BLOCK_SIZE>>>(d_first,d_second,d_weights,d_distance,
                                                         d_next_distance,d_changed,V );
            }

            check_cuda(cudaGetLastError(), "Launch SSSP kernel");

            check_cuda(cudaEventRecord(stop), "Record stop event");

            check_cuda(cudaEventSynchronize(stop), "Synchronize kernel");

            float elapsed = 0.0f;

            check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "Measure kernel time");

            timing.kernel_ms += elapsed;

            int host_changed = 0;

            check_cuda(cudaMemcpy(&host_changed, d_changed,sizeof(int), cudaMemcpyDeviceToHost), "Copy changed flag");

            // Use the newly calculated distances
            int* temp = d_distance;
            d_distance = d_next_distance;
            d_next_distance = temp;

            timing.iterations++;

            // Stop if no distance was improved
            if (host_changed == 0)
                break;
        }

        // Copy final distances to CPU
        vector<int> result(V);

        check_cuda(cudaMemcpy(result.data(), d_distance,V * sizeof(int), cudaMemcpyDeviceToHost), "Copy final distances");

        // Check for reachable negative weight cycles
        bool negative_cycle = false;

        for (const Edge& edge : graph.edges)
        {
            if (result[edge.src] == SSSP_INF)
                continue;

            long long candidate = (long long)result[edge.src] + edge.weight;

            if (candidate < result[edge.dst])
            {
                negative_cycle = true;
                break;
            }
        }

        // Free GPU resources
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_weights);

        cudaFree(d_distance);
        cudaFree(d_next_distance);
        cudaFree(d_changed);

        auto total_end = chrono::high_resolution_clock::now();

        timing.total_ms = chrono::duration<double, std::milli>(total_end - total_start).count();

        if (negative_cycle)
            throw runtime_error("Reachable negative weight cycle detected");

        return result;
    }
    catch (...)
    {
        // Release resources if an error occurs
        if (d_first) cudaFree(d_first);
        if (d_second) cudaFree(d_second);
        if (d_weights) cudaFree(d_weights);

        if (d_distance) cudaFree(d_distance);
        if (d_next_distance) cudaFree(d_next_distance);
        if (d_changed) cudaFree(d_changed);

        throw;
    }
}

// GPU vertex-centric Push
vector<int> sssp_gpu_push(const Graph& graph,int source, SSSPTiming& timing)
{
    return run_sssp_gpu(graph, source, timing, false);
}

// GPU edge-centric
vector<int> sssp_gpu_edge(const Graph& graph, int source, SSSPTiming& timing)
{
    return run_sssp_gpu(graph, source, timing, true);
}
