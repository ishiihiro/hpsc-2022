#include <cstdio>
#include <cstdlib>

__global__ void bucket_sort(int *key, int *bucket, int range){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bucket[i%range] = 0;
    __syncthreads();
    atomicAdd(&bucket[key[i]], 1);
    extern __shared__ int scan[];
    for(int j=1; j<range; j<<=1){
        if(i<range) scan[i] = bucket[i];
        __syncthreads();
        if(i>=j && i<range) bucket[i] += scan[i-j];
        __syncthreads();
    }
    for(int j=0; j<range; j++){
        if(i < bucket[j]){
            key[i] = j;
            return;
        }
    }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucket_sort<<<1, n, range*sizeof(int)>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
}
