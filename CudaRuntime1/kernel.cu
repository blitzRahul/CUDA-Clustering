#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int N = 1000;  // Number of points
const int K = 3;     // Number of clusters
const int DIM = 2;   // Dimension of points
const int MAX_ITER = 100;  // Maximum number of iterations

__global__ void assign_clusters(const float* points, const float* centroids, int* labels, int n, int k, int dim) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float min_dist = INFINITY;
        int best_cluster = 0;
        for (int j = 0; j < k; ++j) {
            float dist = 0;
            for (int d = 0; d < dim; ++d) {
                float diff = points[idx * dim + d] - centroids[j * dim + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[idx] = best_cluster;
    }
}

__global__ void update_centroids(const float* points, const int* labels, float* centroids, int* counts, int n, int k, int dim) {
    extern __shared__ float shared_mem[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    float* shared_centroids = shared_mem;
    int* shared_counts = (int*)&shared_centroids[k * dim];

    if (tid < k * dim) {
        shared_centroids[tid] = 0;
    }
    if (tid < k) {
        shared_counts[tid] = 0;
    }

    __syncthreads();

    if (idx < n) {
        int cluster = labels[idx];
        for (int d = 0; d < dim; ++d) {
            atomicAdd(&shared_centroids[cluster * dim + d], points[idx * dim + d]);
        }
        atomicAdd(&shared_counts[cluster], 1);
    }

    __syncthreads();

    if (tid < k * dim) {
        atomicAdd(&centroids[tid], shared_centroids[tid]);
    }
    if (tid < k) {
        atomicAdd(&counts[tid], shared_counts[tid]);
    }
}

void kmeans(float* h_points, float* h_centroids) {
    float *d_points, *d_centroids;
    int *d_labels, *d_counts;
    cudaMalloc(&d_points, N * DIM * sizeof(float));
    cudaMalloc(&d_centroids, K * DIM * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_points, h_points, N * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;
    int threads = 256;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        assign_clusters<<<blocks, threads>>>(d_points, d_centroids, d_labels, N, K, DIM);
        
        cudaMemset(d_centroids, 0, K * DIM * sizeof(float));
        cudaMemset(d_counts, 0, K * sizeof(int));
        update_centroids<<<blocks, threads, (K * DIM + K) * sizeof(float)>>>(d_points, d_labels, d_centroids, d_counts, N, K, DIM);

        float* h_counts = new float[K];
        cudaMemcpy(h_counts, d_counts, K * sizeof(int), cudaMemcpyDeviceToHost);

        for (int j = 0; j < K; ++j) {
            if (h_counts[j] > 0) {
                for (int d = 0; d < DIM; ++d) {
                    h_centroids[j * DIM + d] /= h_counts[j];
                }
            }
        }

        delete[] h_counts;
        cudaMemcpy(d_centroids, h_centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(h_centroids, d_centroids, K * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}

int main() {
    std::vector<float> points(N * DIM);
    std::vector<float> centroids(K * DIM);

    // Initialize points and centroids with random values
    std::srand(std::time(0));
    for (int i = 0; i < N * DIM; ++i) {
        points[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * DIM; ++i) {
        centroids[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    kmeans(points.data(), centroids.data());

    std::cout << "Final centroids:\n";
    for (int i = 0; i < K; ++i) {
        std::cout << "Centroid " << i << ": ";
        for (int d = 0; d < DIM; ++d) {
            std::cout << centroids[i * DIM + d] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
