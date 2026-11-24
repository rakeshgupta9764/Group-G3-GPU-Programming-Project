#include "graph.h"
#include "bfs.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <exception>
#include <iomanip>

using namespace std;

void print_result( const string& name, const vector<int>& cpu_distance,const vector<int>& gpu_distance, const BFSTiming& timing)
{
    bool correct = (cpu_distance == gpu_distance);

    cout << "\n" << name << '\n';
    cout << "Correctness: "
         << (correct ? "PASSED" : "FAILED") << '\n';

    cout << "Kernel time: "
         << timing.kernel_ms << " ms\n";

    cout << "Total time: "
         << timing.total_ms << " ms\n";

    cout << "Iterations: "
         << timing.iterations << '\n';

    cout << "----------------------------------------\n";
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cout << "Usage: ./bfs_benchmark <graph_file> <source>\n";
        return 1;
    }

    Graph graph;

    if (!load_graph(argv[1], graph))
        return 1;

    int source;

    try
    {
        source = stoi(argv[2]);
    }
    catch (const exception&)
    {
        cerr << "Invalid source vertex\n";
        return 1;
    }

    if (source < 0 || source >= graph.num_vertices)
    {
        cerr << "Source vertex is outside the graph\n";
        return 1;
    }

    cout << "\n========== BFS BENCHMARK ==========\n";

    cout << "Vertices: " << graph.num_vertices << '\n';
    cout << "Edges: " << graph.num_edges << '\n';
    cout << "Source: " << source << '\n';

    cudaDeviceProp prop;

    cudaError_t error = cudaGetDeviceProperties(&prop, 0);

    if (error != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return 1;
    }

    cout << "GPU: " << prop.name << '\n';

    // CPU reference result
    vector<int> cpu_distance = bfs_cpu(graph, source);

    BFSTiming push_timing;
    BFSTiming pull_timing;
    BFSTiming edge_timing;

    try
    {

	// Warm up all GPU implementations
        BFSTiming warmup_timing;

        (void)bfs_gpu_push(graph, source, warmup_timing);
        (void)bfs_gpu_pull(graph, source, warmup_timing);
        (void)bfs_gpu_edge(graph, source, warmup_timing);

        // GPU vertex-centric push
        vector<int> push_distance = bfs_gpu_push(graph, source, push_timing);

        // GPU vertex-centric pull
        vector<int> pull_distance = bfs_gpu_pull(graph, source, pull_timing);

        // GPU edge-centric
        vector<int> edge_distance = bfs_gpu_edge(graph, source, edge_timing);

        cout << "\n========== RESULTS ==========\n";

        print_result(
            "Vertex-centric Push",
            cpu_distance,
            push_distance,
            push_timing);

        print_result(
            "Vertex-centric Pull",
            cpu_distance,
            pull_distance,
            pull_timing);

        print_result(
            "Edge-centric",
            cpu_distance,
            edge_distance,
            edge_timing);

        bool all_correct =
            (cpu_distance == push_distance) &&
            (cpu_distance == pull_distance) &&
            (cpu_distance == edge_distance);

        cout << "\nOverall correctness: "
             << (all_correct ? "PASSED" : "FAILED") << '\n';

        return all_correct ? 0 : 1;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
