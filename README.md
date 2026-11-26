# GPU Graph Algorithm Performance Evaluation

**Course:** CS5658 - GPU Programming  
**Group:** G3

## Team Members
- Adwaith Nath R
- Rakesh Gupta
- Akhilesh Bhoria

## About the Project

In this project, we are implementing and comparing different graph algorithms on the CPU and GPU using CUDA.

We are mainly looking at how different graph representations and parallelization approaches affect the performance of graph algorithms.

The algorithms included in our project are:
- Breadth-First Search (BFS)
- Single-Source Shortest Paths (SSSP)
- PageRank

We have implemented different versions of these algorithms and are comparing their execution times and correctness.

## Graph Representations

We use three different representations for storing graphs:

- **CSR (Compressed Sparse Row)**
- **CSC (Compressed Sparse Column)**
- **COO (Coordinate Format)**

These representations are used depending on the algorithm and the type of traversal being performed.

## BFS Implementations

We implemented three GPU versions of BFS, along with a CPU reference implementation.

| Implementation | Graph Representation |

| CPU BFS | CSR |

| GPU Vertex-centric Push | CSR |

| GPU Vertex-centric Pull | CSC |

| GPU Edge-centric | COO |

All three GPU implementations were tested against the CPU version, and the results matched for our test graphs.

## Dataset

For benchmarking, we use the Wiki-Vote dataset from the Stanford SNAP collection.

Dataset link: https://snap.stanford.edu/data/wiki-Vote.html

The dataset contains:

- Vertices: 7,115
- Directed edges: 103,689

We remapped the original vertex IDs to consecutive IDs so that they can be used easily in our implementation.

For BFS, all edge weights are set to 1.

We have also included a small test graph for checking the implementations.

### Input Format

The graph file starts with the number of vertices and edges, followed by the edge information.

```text
V E
source destination weight
```

## Compilation and Execution

The project requires:

- Linux
- C++17 compiler
- NVIDIA CUDA Toolkit
- CUDA-compatible NVIDIA GPU

We compiled the project using `g++` for the CPU files and `nvcc` for CUDA files.

### BFS Compilation

Run the following commands from the project root:

```bash
g++ -std=c++17 -Wall -Wextra -Iinclude -c src/graph.cpp -o graph.o

g++ -std=c++17 -Wall -Wextra -Iinclude -c src/bfs_cpu.cpp -o bfs_cpu.o

nvcc -std=c++17 -arch=sm_86 -Iinclude \
src/bfs.cu src/bfs_benchmark.cu graph.o bfs_cpu.o \
-o bfs_benchmark
```

### Running BFS

```bash
./bfs_benchmark data/wiki_vote_graph.txt 0
```

The program prints the kernel execution time, total execution time, number of iterations and correctness result.

## BFS Results

**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU  
**Dataset:** Wiki-Vote

The following are the results from our current implementation.

| Implementation | Kernel Time (ms) | Total Time (ms) | Correctness |

| Vertex-centric Push | 0.905056 | 10.0621 | Passed |

| Vertex-centric Pull | 0.546656 | 6.40812 | Passed |

| Edge-centric | 0.208896 | 7.98741 | Passed |

The edge-centric implementation has the lowest kernel time in this run. However, its total execution time is higher than the pull implementation.

This shows that kernel execution time alone does not tell the complete story. The time spent in memory allocation, data transfers and other operations also affects the total time.

These timings were recorded after one warm-up execution of each implementation. We have not performed enough repeated runs to draw a final conclusion about the performance differences.

Kernel time is the accumulated GPU kernel execution time. Total time includes GPU allocation, data transfers, synchronization and cleanup. Graph loading and graph-format construction are excluded.

---

## Single-Source Shortest Paths (SSSP)

For SSSP, we implemented the Bellman-Ford algorithm.

We have a CPU implementation which is used as the reference for checking the GPU results.

### Implementations

| Implementation | Graph Representation |

| CPU Bellman-Ford | Edge List |

| GPU Vertex-centric Push | CSR |

| GPU Edge-centric | COO |

The GPU implementations use synchronous iterations with separate distance arrays. Atomic minimum operations are used to handle cases where multiple threads try to update the distance of the same destination vertex.

### Dataset and Weights

For this experiment, we use the Wiki-Vote graph topology with synthetic edge weights.

- Vertices: 7,115
- Directed edges: 103,689
- Weight range: 1 to 10

The original Wiki-Vote dataset does not contain these weights. We generated them deterministically for our SSSP experiments.

### SSSP Results

**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU

| Implementation | Kernel Time (ms) | Total Time (ms) | Correctness |

| GPU Vertex-centric Push | 2.17466 | 4.48858 | Passed |

| GPU Edge-centric | 0.249792 | 2.3073 | Passed |

Both GPU implementations produced results matching the CPU Bellman-Ford reference for the tested graph.

The edge-centric implementation had lower kernel and total execution times in this run.

These are single-run measurements after GPU warm-up, so more runs are required before making a strong performance comparison.

### Compilation

```bash
nvcc -std=c++17 -arch=sm_86 -Iinclude \
src/graph.cpp src/sssp_cpu.cpp \
src/sssp.cu src/sssp_benchmark.cpp \
-o sssp_benchmark
```

### Running SSSP

For the small test graph:

```bash
./sssp_benchmark data/sssp_tiny_graph.txt 0
```

For Wiki-Vote:

```bash
./sssp_benchmark data/wiki_vote_weighted.txt 0
```

---

## PageRank

We have also implemented PageRank on both CPU and GPU.

For the GPU implementation, we tried two approaches:

- Vertex-centric Push using CSR.
- Vertex-centric Pull using CSC.

### Implementations

| Implementation | Graph Representation |

| CPU Reference | CSR |

| GPU Vertex-centric Push | CSR |

| GPU Vertex-centric Pull | CSC |

### Parameters Used

| Parameter | Value |

| Damping factor | 0.85 |

| Convergence tolerance | 0.000001 |

| Maximum iterations | 100 |

| Initial rank of each vertex | 1/V |

We also handled dangling vertices. These are vertices which do not have any outgoing edges.

Their rank contribution is redistributed equally among all vertices during the PageRank computation.

### PageRank Results

**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU  
**Dataset:** Wiki-Vote

| Implementation | Kernel Time (ms) | Total Time (ms) | Correctness |

| GPU Push | 10.0729 | 12.6024 | Passed |

| GPU Pull | 10.2363 | 12.9245 | Passed |

Both implementations were compared with the CPU reference. The L1 error was below 0.000001 for both versions.

The sum of PageRank values was approximately 0.999986.

The results are from single runs after warm-up. More repeated measurements are needed for a proper performance comparison.

### Compilation

```bash
nvcc -std=c++17 -arch=sm_86 -Iinclude \
src/graph.cpp src/pagerank_cpu.cpp \
src/pagerank.cu src/pagerank_benchmark.cpp \
-o pagerank_benchmark
```

### Running PageRank

For the small graph:

```bash
./pagerank_benchmark data/tiny_graph.txt
```

For Wiki-Vote:

```bash
./pagerank_benchmark data/wiki_vote_graph.txt
```

---

## Current Status

So far, we have implemented and tested BFS, SSSP and PageRank using different approaches.

The GPU implementations have been checked against the corresponding CPU reference implementations, and the tested results matched within the specified correctness criteria.

We have also collected initial timing results on the RTX 3050 Laptop GPU. These results give us an initial idea of how the different implementations perform, but further repeated runs and profiling would be needed to make more reliable comparisons.

The main focus of the project is to understand how graph representation and the way we distribute work among GPU threads affect the execution of graph algorithms.
