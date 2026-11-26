
#include "graph.h"
#include "pagerank.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <exception>

using namespace std;

bool print_result(const string& name,const vector<float>& cpu_rank,const vector<float>& gpu_rank,const PageRankTiming& timing)
{
    float l1_error = 0.0f;
    float max_error = 0.0f;

    for (int v = 0; v < (int)cpu_rank.size(); v++)
    {
        float error = fabs(cpu_rank[v] - gpu_rank[v]);

        l1_error += error;

        if (error > max_error)
            max_error = error;
    }

    float rank_sum = accumulate(gpu_rank.begin(), gpu_rank.end(), 0.0f);

    // Allow small floating-point differences.
    bool correct = (l1_error < 0.0001f);

    cout << "\n" << name << '\n';

    cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << '\n';

    cout << "L1 error: " << l1_error << '\n';
    cout << "Maximum error: " << max_error << '\n';
    cout << "Total rank: " << rank_sum << '\n';

    cout << "Kernel time: " << timing.kernel_ms << " ms\n";
    cout << "Total time: " << timing.total_ms << " ms\n";
    cout << "Iterations: " << timing.iterations << '\n';

    cout << "----------------------------------------\n";

    return correct;
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        cout << "Usage: ./pagerank_benchmark <graph_file>\n";
        return 1;
    }

    Graph graph;

    if (!load_graph(argv[1], graph))
        return 1;

    try
    {
        // CPU reference
        int cpu_iterations = 0;

        vector<float> cpu_rank = pagerank_cpu(graph, cpu_iterations);

        // GPU information
        cudaDeviceProp prop;

        cudaError_t error = cudaGetDeviceProperties(&prop, 0);

        if (error != cudaSuccess)
        {
            cerr << cudaGetErrorString(error) << '\n';
            return 1;
        }

        // Warm up both GPU implementations.
        PageRankTiming warmup_timing;

        (void)pagerank_gpu_push(graph, warmup_timing);
        (void)pagerank_gpu_pull(graph, warmup_timing);

        // Actual measurements
        PageRankTiming push_timing;
        PageRankTiming pull_timing;

        vector<float> push_rank = pagerank_gpu_push(graph, push_timing);

        vector<float> pull_rank = pagerank_gpu_pull(graph, pull_timing);

        cout << "\n-------------------PAGERANK BENCHMARK -------------------\n";

        cout << "Vertices: " << graph.num_vertices << '\n';
        cout << "Edges: " << graph.num_edges << '\n';
        cout << "GPU: " << prop.name << '\n';

        cout << "\nCPU iterations: " << cpu_iterations << '\n';

        bool push_correct = print_result( "GPU Vertex-centric Push",cpu_rank,push_rank,push_timing);

        bool pull_correct = print_result("GPU Vertex-centric Pull",cpu_rank, pull_rank,pull_timing);

        // Display individual scores only for small graphs.
        if (graph.num_vertices <= 20)
        {
            cout << "\nVertex PageRank:\n";

            for (int v = 0; v < graph.num_vertices; v++)
            {
                cout << "Vertex " << v << " -> CPU: " << cpu_rank[v] << ", Push: " << push_rank[v] << ", Pull: " << pull_rank[v]
                     << '\n';
            }
        }

        bool all_correct = push_correct && pull_correct;

        cout << "\nOverall correctness: " << (all_correct ? "PASSED" : "FAILED") << '\n';

        return all_correct ? 0 : 1;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
