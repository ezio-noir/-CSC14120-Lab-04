#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

__device__ int started_block_count = 0;
__device__ int completed_block_count = 0;

__global__ void g_extractBits(uint32_t* in, int n, int k, int* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (in[idx] >> k) & 1;
    }
}

__global__ void g_scan(int* in, int n, int* out, int* block_sums) {
    extern __shared__ int s_data[];
    __shared__ int prev_sum;

    if (threadIdx.x == 0) {
        s_data[0] = atomicAdd(&started_block_count, 1);
    }
    __syncthreads();

    int block_id = s_data[0];
    int i1 = block_id * 2 * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x;
    if (i1 < n) s_data[threadIdx.x] = in[i1];
    if (i2 < n) s_data[threadIdx.x + blockDim.x] = in[i2];
    __syncthreads();

    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
        int s_data_idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (s_data_idx < 2 * blockDim.x) {
            s_data[s_data_idx] += s_data[s_data_idx - stride];
        }
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        int s_data_idx = (threadIdx.x + 1) * 2 * stride - 1 + stride;
        if (s_data_idx < 2 * blockDim.x) {
            s_data[s_data_idx] += s_data[s_data_idx - stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sums[block_id] = s_data[2 * blockDim.x - 1];
        if (block_id > 0) {
            while (atomicAdd(&completed_block_count, 0) < block_id) {}
            prev_sum = block_sums[block_id - 1];
            block_sums[block_id] += prev_sum;
            __threadfence();
        }
        atomicAdd(&completed_block_count, 1);
    }
    __syncthreads();

    if (block_id > 0) {
        if (i1 + 1 < n) out[i1 + 1] = s_data[threadIdx.x] + prev_sum;
        if (i2 + 1 < n) out[i2 + 1] = s_data[threadIdx.x + blockDim.x] + prev_sum;
    } else {
        if (i1 + 1 < n) out[i1 + 1] = s_data[threadIdx.x];
        if (i2 + 1 < n) out[i2 + 1] = s_data[threadIdx.x + blockDim.x];
    }
}

__global__ void g_computeRankAndAssign(uint32_t* in, int* bits, int* num_ones_before, int n, uint32_t* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int num_zeros = n - num_ones_before[n - 1] - bits[n - 1];
        int rank = (bits[idx] == 0) ? (idx - num_ones_before[idx]) : (num_zeros + num_ones_before[idx]);
        out[rank] = in[idx];
    }
}

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    int zero = 0;

    uint32_t* d_src;
    uint32_t* d_dst;
    int* d_bits;
    int* d_num_ones_berfore;
    int* d_block_sums;

    CHECK(cudaMalloc((void**)&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc((void**)&d_dst, n * sizeof(uint32_t)));
    CHECK(cudaMalloc((void**)&d_bits, n * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_num_ones_berfore, n * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_block_sums, (n + blockSize * 2 - 1) / (blockSize * 2) * sizeof(int)));

    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_num_ones_berfore + 0, 0, sizeof(int)));

    for (int bit_idx = 0; bit_idx < sizeof(uint32_t) * 8; ++bit_idx)
    {
        // Extract bits
        g_extractBits<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_src, n, bit_idx, d_bits);

        // Compute nOnesBefore
        CHECK(cudaMemcpyToSymbol(started_block_count, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(completed_block_count, &zero, sizeof(int)));
        g_scan<<<(n + blockSize * 2 - 1) / (blockSize * 2), blockSize, 2 * blockSize * sizeof(int)>>>(d_bits, n, d_num_ones_berfore, d_block_sums);

        // Compute rank and write to dst
        g_computeRankAndAssign<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_src, d_bits, d_num_ones_berfore, n, d_dst);
        
        // Swap src and dst
        uint32_t * temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }

    // Does out array contain results?
    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_num_ones_berfore));
    CHECK(cudaFree(d_block_sums));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
