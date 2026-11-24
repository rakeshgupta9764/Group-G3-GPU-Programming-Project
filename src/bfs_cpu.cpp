#include "bfs.h"

#include <queue>
#include <stdexcept>
#include <vector>

using namespace std;

vector<int> bfs_cpu(const Graph& graph, int source)
{

    int V = graph.num_vertices;

    // Validate source vertex.
    if (source < 0 || source >= V)
    {
        throw invalid_argument("Invalid BFS source vertex");
    }

    //Initially all vertices are unvisited.
    vector<int> distance(V, -1);

    // Fifo queue for BFS traversal.
    queue<int> q;

    //Start BFS from the source.
    distance[source] = 0;
    q.push(source);

    while (!q.empty())
    {

        int u = q.front();
        q.pop();

        //Visit all outgoing neighbours of u using CSR.
        int start = graph.csr.row_ptr[u];
        int end = graph.csr.row_ptr[u + 1];

        for (int e = start; e < end; e++)
	{

            int v = graph.csr.col_idx[e];

            // Discover v only if it has not been visited.
            if (distance[v] == -1)
	    {

                distance[v] = distance[u] + 1;

                q.push(v);
            }
        }
    }

    return distance;
}
