
#include "pagerank.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace std;

vector<float> pagerank_cpu(const Graph& graph, int& iterations)
{
    int V = graph.num_vertices;

    if (V <= 0)
        throw invalid_argument("Graph is empty");

    float damping = PAGERANK_DAMPING;

    // Initially, every vertex has equal rank.
    vector<float> rank(V, 1.0f / V);
    vector<float> next_rank(V);

    iterations = 0;

    for (int i = 0; i < PAGERANK_MAX_ITERATIONS; i++)
    {
        // Find the total rank of dangling vertices.
        float dangling_sum = 0.0f;

        for (int u = 0; u < V; u++)
        {
            if (graph.out_degree[u] == 0)
                dangling_sum += rank[u];
        }

        // Base rank for every vertex.
        float base = (1.0f - damping) / V + damping * dangling_sum / V;

        fill(next_rank.begin(), next_rank.end(), base);

        // Vertex-centric Push
        for (int u = 0; u < V; u++)
        {
            int degree = graph.out_degree[u];

            if (degree == 0)
                continue;

            float contribution =
                damping * rank[u] / degree;

            // Distribute rank to outgoing neighbours.
            for (int e = graph.csr.row_ptr[u]; e < graph.csr.row_ptr[u + 1]; e++)
            {
                int v = graph.csr.col_idx[e];

                next_rank[v] += contribution;
            }
        }

        // Calculate total rank difference.
        float difference = 0.0f;

        for (int v = 0; v < V; v++)
        {
            difference += fabs(next_rank[v] - rank[v]);
        }

        rank.swap(next_rank);

        iterations++;

        // Stop when PageRank converges.
        if (difference < PAGERANK_TOLERANCE)
            break;
    }

    return rank;
}
