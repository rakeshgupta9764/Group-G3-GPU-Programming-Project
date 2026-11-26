#include "pagerank.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// Check CUDA errors
void check_pagerank_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw runtime_error(string(operation) + ": " + cudaGetErrorString(error));
    }
}

// Initialize the next PageRank array.
// Also calculate rank contributed by dangling vertices.
__global__ void pagerank_initialize_kernel(const float* rank, const int* out_degree,float* next_rank,int V,float damping)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    float dangling_sum = 0.0f;

    for (int u = 0; u < V; u++)
    {
        if (out_degree[u] == 0)
            dangling_sum += rank[u];
    }

    float base = (1.0f - damping) / V + damping * dangling_sum / V;

    for (int v = 0; v < V; v++)
        next_rank[v] = base;
}

// GPU vertex-centric Push using CSR
__global__ void pagerank_push_kernel(const int* row_ptr, const int* col_idx, const int* out_degree, const float* rank, float* next_rank, int V,float damping)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V)
        return;

    int degree = out_degree[u];

    if (degree == 0)
        return;

    float contribution = damping * rank[u] / degree;

    for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++)
    {
        int v = col_idx[e];

        atomicAdd(&next_rank[v], contribution);
    }
}

// GPU vertex-centric Pull using CSC
__global__ void pagerank_pull_kernel(const int* col_ptr,const int* row_idx, const int* out_degree, const float* rank, float* next_rank, int V, float damping)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= V)
        return;

    float incoming_sum = 0.0f;

    // Collect contributions from incoming neighbours.
    for (int e = col_ptr[v]; e < col_ptr[v + 1]; e++)
    {
        int u = row_idx[e];

        if (out_degree[u] > 0)
        {
            incoming_sum += rank[u] / out_degree[u];
        }
    }

    next_rank[v] += damping * incoming_sum;
}

// Calculate convergence using L1 difference.
__global__ void pagerank_convergence_kernel( const float* rank, const float* next_rank,int* converged,int V,float tolerance)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    float difference = 0.0f;

    for (int v = 0; v < V; v++)
    {
        difference += fabsf(next_rank[v] - rank[v]);
    }

    *converged = (difference < tolerance) ? 1 : 0;
}

// Common GPU PageRank execution function
vector<float> run_pagerank_gpu(const Graph& graph,PageRankTiming& timing, bool pull)
{
    timing = PageRankTiming{};

    int V = graph.num_vertices;
    int E = graph.num_edges;

    if (V <= 0)
        throw invalid_argument("Graph is empty");

    auto total_start = chrono::high_resolution_clock::now();

    int* d_first = nullptr;
    int* d_second = nullptr;
    int* d_out_degree = nullptr;

    float* d_rank = nullptr;
    float* d_next_rank = nullptr;

    int* d_converged = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    try
    {
        // Allocate graph arrays
        check_pagerank_cuda(cudaMalloc(&d_first, (V + 1) * sizeof(int)), "Allocate graph pointers");

        if (E > 0)
        {
            check_pagerank_cuda(cudaMalloc(&d_second, E * sizeof(int)), "Allocate graph neighbours");
        }

        check_pagerank_cuda(cudaMalloc(&d_out_degree, V * sizeof(int)), "Allocate out-degree");

        // Copy CSR or CSC graph data
        if (pull)
        {
            check_pagerank_cuda(cudaMemcpy(d_first,graph.csc.col_ptr.data(),(V + 1) * sizeof(int),cudaMemcpyHostToDevice), "Copy CSC pointers");

            if (E > 0)
            {
                check_pagerank_cuda(cudaMemcpy(d_second,graph.csc.row_idx.data(),E * sizeof(int),cudaMemcpyHostToDevice), "Copy CSC neighbours");
            }
        }
        else
        {
            check_pagerank_cuda(cudaMemcpy(d_first,graph.csr.row_ptr.data(),(V + 1) * sizeof(int),cudaMemcpyHostToDevice), "Copy CSR pointers");

            if (E > 0)
            {
                check_pagerank_cuda(cudaMemcpy(d_second,graph.csr.col_idx.data(),E * sizeof(int),cudaMemcpyHostToDevice), "Copy CSR neighbours");
            }
        }

        check_pagerank_cuda(cudaMemcpy(d_out_degree,graph.out_degree.data(),V * sizeof(int),cudaMemcpyHostToDevice),"Copy out-degree");

        // Allocate PageRank arrays
        check_pagerank_cuda(cudaMalloc(&d_rank, V * sizeof(float)), "Allocate rank");

        check_pagerank_cuda(cudaMalloc(&d_next_rank, V * sizeof(float)), "Allocate next rank");

        check_pagerank_cuda(cudaMalloc(&d_converged, sizeof(int)), "Allocate convergence flag");

        // Initial rank = 1 / V
        vector<float> initial_rank(V, 1.0f / V);

        check_pagerank_cuda(cudaMemcpy(d_rank,initial_rank.data(),V * sizeof(float),cudaMemcpyHostToDevice), "Initialize PageRank");

        check_pagerank_cuda(cudaEventCreate(&start_event), "Create start event");

        check_pagerank_cuda(cudaEventCreate(&stop_event), "Create stop event");

        const int BLOCK_SIZE = 256;
        int blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // PageRank iterations
        for (int iteration = 0; iteration < PAGERANK_MAX_ITERATIONS; iteration++)
        {
            check_pagerank_cuda( cudaEventRecord(start_event), "Record start event");

            // Calculate dangling mass and initialize next rank.
            pagerank_initialize_kernel<<<1, 1>>>(d_rank,d_out_degree,d_next_rank,V,PAGERANK_DAMPING);

            check_pagerank_cuda(cudaGetLastError(),"Launch initialization kernel");

            if (pull)
            {
                // Vertex-centric Pull
                pagerank_pull_kernel<<<blocks, BLOCK_SIZE>>>(d_first,d_second,d_out_degree,d_rank,d_next_rank,V,PAGERANK_DAMPING);
            }
            else
            {
                // Vertex-centric Push
                pagerank_push_kernel<<<blocks, BLOCK_SIZE>>>(d_first,d_second,d_out_degree,d_rank,d_next_rank,V,PAGERANK_DAMPING);
            }

            check_pagerank_cuda(cudaGetLastError(),"Launch PageRank kernel");

            // Check convergence.
            pagerank_convergence_kernel<<<1, 1>>>(d_rank,d_next_rank,d_converged,V,PAGERANK_TOLERANCE);

            check_pagerank_cuda(cudaGetLastError(), "Launch convergence kernel");

            check_pagerank_cuda(cudaEventRecord(stop_event), "Record stop event");

            check_pagerank_cuda(cudaEventSynchronize(stop_event),"Synchronize PageRank");

            float elapsed = 0.0f;

            check_pagerank_cuda(cudaEventElapsedTime(&elapsed,start_event,stop_event),"Measure PageRank time");

            timing.kernel_ms += elapsed;

            int host_converged = 0;

            check_pagerank_cuda(cudaMemcpy(&host_converged, d_converged,sizeof(int),cudaMemcpyDeviceToHost), "Copy convergence flag");

            // Swap the two rank arrays.
            float* temp = d_rank;
            d_rank = d_next_rank;
            d_next_rank = temp;

            timing.iterations++;

            if (host_converged == 1)
                break;
        }

        // Copy final PageRank to CPU.
        vector<float> result(V);

        check_pagerank_cuda(cudaMemcpy(result.data(),d_rank,V * sizeof(float),cudaMemcpyDeviceToHost), "Copy final PageRank");

        // Clean up
        cudaEventDestroy(start_event);
        start_event = nullptr;

        cudaEventDestroy(stop_event);
        stop_event = nullptr;

        cudaFree(d_first);
        d_first = nullptr;

        cudaFree(d_second);
        d_second = nullptr;

        cudaFree(d_out_degree);
        d_out_degree = nullptr;

        cudaFree(d_rank);
        d_rank = nullptr;

        cudaFree(d_next_rank);
        d_next_rank = nullptr;

        cudaFree(d_converged);
        d_converged = nullptr;

        auto total_end = chrono::high_resolution_clock::now();

        timing.total_ms = chrono::duration<double, milli>( total_end - total_start).count();

        return result;
    }
    catch (...)
    {
        // Release allocated resources if an error occurs.
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);

        if (d_first) cudaFree(d_first);
        if (d_second) cudaFree(d_second);
        if (d_out_degree) cudaFree(d_out_degree);

        if (d_rank) cudaFree(d_rank);
        if (d_next_rank) cudaFree(d_next_rank);

        if (d_converged) cudaFree(d_converged);

        throw;
    }
}

// GPU vertex-centric Push
vector<float> pagerank_gpu_push( const Graph& graph, PageRankTiming& timing)
{
    return run_pagerank_gpu(graph, timing, false);
}

// GPU vertex-centric Pull
vector<float> pagerank_gpu_pull( const Graph& graph, PageRankTiming& timing)
{
    return run_pagerank_gpu(graph, timing, true);
}
