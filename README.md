
# GPU Graph Algorithm Performance Evaluation

**Course:** CS5658 - GPU Programming
**Group:** G3

## Team Members
- Adwaith Nath R
- Rakesh Gupta
- Akhilesh Bhoria

## Project Overview

This project evaluates different GPU implementations of graph algorithms using CUDA.

Algorithms planned:
- Breadth-First Search (BFS)
- Single-Source Shortest Paths (SSSP), using Bellman-Ford
- PageRank

## Current Implementation Status

### Graph Representations
- CSR (Compressed Sparse Row)
- CSC (Compressed Sparse Column)
- COO (Coordinate Format)

### BFS Implementations

Implementation | Representation
CPU reference | CSR |
GPU vertex-centric Push | CSR |
GPU vertex-centric Pull | CSC |
GPU edge-centric | COO |

All three GPU implementations have been validated against the CPU reference.

PageRank implementation is not yet included.

## Dataset

The medium-sized benchmark uses the Wiki-Vote dataset from Stanford SNAP.

Source: https://snap.stanford.edu/data/wiki-Vote.html

- Vertices: 7,115
- Directed edges: 103,689

Original vertex IDs are remapped to consecutive IDs. All edge weights are set to 1.

A small test graph is also included.

## Input Format

The graph file begins with the number of vertices and edges, followed by directed edges:

    V E
    source destination weight

## Requirements

- Linux
- C++17 compiler
- NVIDIA CUDA Toolkit
- CUDA-capable NVIDIA GPU

## Compilation

Run from the project root:

    g++ -std=c++17 -Wall -Wextra -Iinclude -c src/graph.cpp -o graph.o

    g++ -std=c++17 -Wall -Wextra -Iinclude -c src/bfs_cpu.cpp -o bfs_cpu.o

    nvcc -std=c++17 -arch=sm_86 -Iinclude src/bfs.cu src/bfs_benchmark.cu graph.o bfs_cpu.o -o bfs_benchmark

The `sm_86` architecture flag was used for the NVIDIA RTX 3050 Laptop GPU.

## Running

    ./bfs_benchmark data/wiki_vote_graph.txt 0

The program reports kernel time, total execution time, iteration count and correctness.

## Preliminary BFS Results

GPU: NVIDIA GeForce RTX 3050 Laptop GPU

Dataset: Wiki-Vote

 Implementation      | Kernel time (ms) | Total time (ms) | Correctness
 
 Vertex-centric Push | 0.905056 | 10.0621 | PASSED |
 
 Vertex-centric Pull | 0.546656 | 6.40812 | PASSED |
 
 Edge-centric        | 0.208896 | 7.98741 | PASSED |

These are preliminary single-run measurements after one warm-up execution of each implementation. They are not statistically conclusive.

Kernel time is accumulated GPU kernel execution time. Total time includes GPU allocation, transfers, synchronization and cleanup. Graph loading and graph-format construction are excluded.


## Single-Source Shortest Paths (SSSP)

SSSP is implemented using the Bellman-Ford algorithm.

### Implementations

Implementation | Graph representation |


CPU Bellman-Ford reference | Edge list |

GPU Vertex-centric Push | CSR |

GPU Edge-centric | COO |

The GPU implementations use synchronous iterations with separate distance arrays. Atomic minimum operations handle concurrent updates to destination vertices.

### Weighted Dataset

The SSSP benchmark uses the Wiki-Vote graph topology with deterministic synthetic edge weights from 1 to 10.

- Vertices: 7,115
- Directed edges: 103,689
- Weight range: 1 to 10

These weights are synthetic and are not part of the original Wiki-Vote dataset.

### Preliminary SSSP Results

GPU: NVIDIA GeForce RTX 3050 Laptop GPU

Implementation | Kernel time (ms) | Total time (ms) | Correctness |


GPU Vertex-centric Push | 2.17466 | 4.48858 | PASSED |

GPU Edge-centric | 0.249792 | 2.3073 | PASSED |

Both GPU implementations matched the CPU Bellman-Ford reference.

These are preliminary measurements from a single run after GPU warm-up, not statistically conclusive performance results.

### Compilation and Execution

    nvcc -std=c++17 -arch=sm_86 -Iinclude \
    src/graph.cpp src/sssp_cpu.cpp \
    src/sssp.cu src/sssp_benchmark.cpp \
    -o sssp_benchmark

    ./sssp_benchmark data/sssp_tiny_graph.txt 0

    ./sssp_benchmark data/wiki_vote_weighted.txt 0
