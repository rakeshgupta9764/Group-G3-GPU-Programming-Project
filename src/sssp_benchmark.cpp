#include "graph.h"
#include "sssp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <exception>

using namespace std;

void print_result(const string& name, const vector<int>& cpu_distance, const vector<int>& gpu_distance,const SSSPTiming& timing)
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
        cout << "Usage: ./sssp_benchmark <graph_file> <source>\n";
        return 1;
    }

    Graph graph;

    if (!load_graph(argv[1], graph))
        return 1;

    int source = stoi(argv[2]);

    try
    {
        // CPU reference
        int cpu_iterations = 0;

        vector<int> cpu_distance =
            sssp_cpu_bellman_ford(graph, source, cpu_iterations);

        // GPU warm-up
        SSSPTiming warmup_timing;

        (void)sssp_gpu_push(graph, source, warmup_timing);
        (void)sssp_gpu_edge(graph, source, warmup_timing);

        // Actual GPU measurements
        SSSPTiming push_timing;
        SSSPTiming edge_timing;

        vector<int> push_distance = sssp_gpu_push(graph, source, push_timing);

        vector<int> edge_distance = sssp_gpu_edge(graph, source, edge_timing);

        cout << "\n-------------SSSP BENCHMARK ---------------\n";

        cout << "Vertices: " << graph.num_vertices << '\n';
        cout << "Edges: " << graph.num_edges << '\n';
        cout << "Source: " << source << '\n';

        cudaDeviceProp prop;

        cudaError_t error = cudaGetDeviceProperties(&prop, 0);

        if (error == cudaSuccess)
            cout << "GPU: " << prop.name << '\n';

        cout << "\n--------------- RESULTS -----------------\n";

        cout << "CPU Bellman-Ford iterations: " << cpu_iterations << '\n';

        print_result("GPU Vertex-centric Push", cpu_distance, push_distance,push_timing);

        print_result("GPU Edge-centric",cpu_distance,edge_distance, edge_timing);

        // Print individual distances for small graphs only
        if (graph.num_vertices <= 20)
        {
            cout << "\nVertex distances:\n";

            for (int v = 0; v < graph.num_vertices; v++)
            {
                cout << "Vertex " << v << " -> CPU: " << cpu_distance[v] << ", Push: " << push_distance[v] << ", Edge: " << edge_distance[v] << '\n';
            }
        }

        bool all_correct = (cpu_distance == push_distance) && (cpu_distance == edge_distance);

        cout << "\nOverall correctness: " << (all_correct ? "PASSED" : "FAILED") << '\n';

        return all_correct ? 0 : 1;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
