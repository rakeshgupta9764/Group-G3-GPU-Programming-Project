#include "sssp.h"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

vector<int> sssp_cpu_bellman_ford(const Graph& graph, int source, int& iterations)
{
    int V = graph.num_vertices;

    if (source < 0 || source >= V)
        throw invalid_argument("Invalid source vertex");

    vector<int> distance(V, SSSP_INF);

    distance[source] = 0;

    iterations = 0;

    // BellmanFord at most V-1 iterations
    for (int i = 0; i < V - 1; i++)
    {
        vector<int> next_distance = distance;

        bool changed = false;

        // Examine every edge
        for (const Edge& edge : graph.edges)
        {
            int u = edge.src;
            int v = edge.dst;
            int weight = edge.weight;

            // Ignore unreachable vertices
            if (distance[u] == SSSP_INF)
                continue;

            long long new_distance = (long long)distance[u] + weight;

            if (new_distance < next_distance[v])
            {
                next_distance[v] = (int)new_distance;
                changed = true;
            }
        }

        distance.swap(next_distance);

        iterations++;

        // Stop if no distance was updated
        if (!changed)
            break;
    }

    // Detect a reachable negative-weight cycle
    for (const Edge& edge : graph.edges)
    {
        if (distance[edge.src] == SSSP_INF)
            continue;

        long long new_distance = (long long)distance[edge.src] + edge.weight;

        if (new_distance < distance[edge.dst])
        {
            throw runtime_error("Reachable negative weight cycle detected");
        }
    }

    return distance;
}
