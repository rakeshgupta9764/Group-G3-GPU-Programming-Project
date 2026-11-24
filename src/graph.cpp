#include "graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>

using namespace std;

// Read an edge-list file.
bool load_graph(const string& filename, Graph& graph)
{

    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error: Cannot open " << filename << '\n';
        return false;
    }

    Graph temp;
    string line;

    bool header_found = false;
    int edge_count = 0;
    int line_number = 0;

    while (getline(file, line))
    {

        line_number++;

        size_t first = line.find_first_not_of(" \t\r");

        if (first == string::npos || line[first] == '#')
            continue;

        istringstream iss(line);

        // Read the graph header.
        if (!header_found)
	{

            string extra;

            if (!(iss >> temp.num_vertices >> temp.num_edges) || (iss >> extra))
	    {

                cerr << "Error: Invalid graph header\n";
                return false;
            }

            if (temp.num_vertices <= 0 || temp.num_edges < 0)
	    {
                cerr << "Error: Invalid vertex/edge count\n";
                return false;
            }

            temp.edges.reserve(temp.num_edges);
            header_found = true;
        }

        // Read individual edges.
        else
	{

            Edge edge;

            if (!(iss >> edge.src >> edge.dst))
	    {
                cerr << "Error: Invalid edge at line "
                          << line_number << '\n';
                return false;
            }

            // Edge weight is optional.
            // If omitted, use weight 1.
            edge.weight = 1;

            iss >> ws;

            if (!iss.eof())
	    {

                if (!(iss >> edge.weight))
		{
                    cerr << "Error: Invalid weight at line "
                              << line_number << '\n';
                    return false;
                }

                iss >> ws;

                if (!iss.eof())
		{
                    cerr << "Error: Extra data at line "
                              << line_number << '\n';
                    return false;
                }
            }

            // Validate vertex IDs.
            if (edge.src < 0 || edge.src >= temp.num_vertices || edge.dst < 0 || edge.dst >= temp.num_vertices)
	    {

                cerr << "Error: Invalid vertex ID at line "
                          << line_number << '\n';
                return false;
            }

            temp.edges.push_back(edge);
            edge_count++;
        }
    }

    if (!header_found)
    {
        cerr << "Error: Missing graph header\n";
        return false;
    }

    // Verify the number of edges.
    if (edge_count != temp.num_edges)
    {
        cerr << "Error: Expected " << temp.num_edges << " edges but found " << edge_count << '\n';
        return false;
    }

    // Store outgoing degree of every vertex.
    temp.out_degree.assign(temp.num_vertices, 0);

    for (const Edge& edge : temp.edges)
    {
        temp.out_degree[edge.src]++;
    }

    // Construct all three representations.
    build_graph_formats(temp);

    graph = move(temp);

    return true;
}


// Build CSR, CSC and COO representations.
void build_graph_formats(Graph& graph)
{

    int V = graph.num_vertices;
    int E = graph.num_edges;

    // ----------------------------------
    // COO REPRESENTATION
    // ----------------------------------

    graph.coo.src.resize(E);
    graph.coo.dst.resize(E);
    graph.coo.weights.resize(E);

    for (int i = 0; i < E; i++)
    {

        graph.coo.src[i] = graph.edges[i].src;
        graph.coo.dst[i] = graph.edges[i].dst;
        graph.coo.weights[i] = graph.edges[i].weight;
    }


    // -------------------------------------
    // CSR REPRESENTATION
    // -------------------------------------

    graph.csr.row_ptr.assign(V + 1, 0);

    // Count outgoing edges.
    for (const Edge& edge : graph.edges)
    {
        graph.csr.row_ptr[edge.src + 1]++;
    }

    // Prefix sum.
    for (int i = 1; i <= V; i++)
    {
        graph.csr.row_ptr[i] += graph.csr.row_ptr[i - 1];
    }

    graph.csr.col_idx.resize(E);
    graph.csr.weights.resize(E);

    // Temporary positions for filling adjacency lists.
    vector<int> position = graph.csr.row_ptr;

    for (const Edge& edge : graph.edges) {

        int p = position[edge.src]++;

        graph.csr.col_idx[p] = edge.dst;
        graph.csr.weights[p] = edge.weight;
    }


    // ----------------------------------
    // CSC REPRESENTATION
    // ----------------------------------

    graph.csc.col_ptr.assign(V + 1, 0);

    // Count incoming edges.
    for (const Edge& edge : graph.edges)
    {
        graph.csc.col_ptr[edge.dst + 1]++;
    }

    // Prefix sum.
    for (int i = 1; i <= V; i++)
    {
        graph.csc.col_ptr[i] += graph.csc.col_ptr[i - 1];
    }

    graph.csc.row_idx.resize(E);
    graph.csc.weights.resize(E);

    // Temporary positions for filling incoming adjacency lists.
    position = graph.csc.col_ptr;

    for (const Edge& edge : graph.edges)
    {

        int p = position[edge.dst]++;

        graph.csc.row_idx[p] = edge.src;
        graph.csc.weights[p] = edge.weight;
    }
}


// Display graph information.
void print_graph_summary(const Graph& graph)
{

    int max_degree = 0;
    long long total_degree = 0;

    for (int degree : graph.out_degree)
    {

        max_degree = max(max_degree, degree);
        total_degree += degree;
    }

    double average_degree =
        static_cast<double>(total_degree) / graph.num_vertices;

    cout << "\n------------- GRAPH SUMMARY -------------------\n";

    cout << "Vertices       : " << graph.num_vertices << '\n';
    cout << "Edges          : " << graph.num_edges << '\n';

    cout << "Average degree : " << average_degree << '\n';
    cout << "Maximum degree : " << max_degree << '\n';

    cout << "\nCSR arrays     : " << graph.csr.col_idx.size() << '\n';

    cout << "CSC arrays     : " << graph.csc.row_idx.size() << '\n';

    cout << "COO arrays     : " << graph.coo.src.size() << '\n';

    cout << "---------------------------------------------------\n";
}
