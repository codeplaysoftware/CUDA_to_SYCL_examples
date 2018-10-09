#include <stdio.h>                                                              
#include <vector>                                                               
                                                                                
// CUDA device kernel                                                           
__global__ void vector_add(const float *A, const float *B, float *C,            
                           size_t array_size) {                                 
  // local thread id                                                            
  size_t id = threadIdx.x;                                                      
  // calculating global id                                                      
  size_t total_threads = gridDim.x * blockDim.x;                                
  for (size_t i = id; i < array_size; i += total_threads) {                     
    C[i] = A[i] + B[i];                                                         
  }                                                                             
}                                                                               
                                                                                
int main() {                                                                    
  const size_t array_size = 256;                                                
  std::vector<float> A(array_size, 1.0f);                                       
  std::vector<float> B(array_size, 1.0f);                                       
  std::vector<float> C(array_size);                                             
                                                                                
  // allocating device memory                                                   
  float *A_dev;                                                                 
  float *B_dev;                                                                 
  float *C_dev;                                                                 
  cudaMalloc((void **)&A_dev, array_size * sizeof(float));                      
  cudaMalloc((void **)&B_dev, array_size * sizeof(float));                      
  cudaMalloc((void **)&C_dev, array_size * sizeof(float));                      
                                                                                
  // explicitly copying data from host to device                                
  cudaMemcpy(A_dev, A.data(), array_size * sizeof(float),                       
             cudaMemcpyHostToDevice);                                           
  cudaMemcpy(B_dev, B.data(), array_size * sizeof(float),                       
             cudaMemcpyHostToDevice);                                           
                                                                                
  // getting device property in order to query device parameters                
  cudaDeviceProp prop;                                                          
  cudaGetDeviceProperties(&prop, 0);                                                                         
  const size_t max_thread_per_block = prop.maxThreadsPerBlock;                  
  const size_t num_thread_per_block =                                           
      std::min(max_thread_per_block, array_size);                               
  const size_t num_block_per_grid =                                                                                                  
               (size_t)std::ceil(((float)array_size) / num_thread_per_block);  
  // constructing block size                                                    
  dim3 block_size(num_thread_per_block, 1, 1);                                  
  // constructing number of blocks (grid size)                                  
  dim3 num_blocks(num_block_per_grid, 1, 1);                                    
  // launching and executing cuda kernel                                        
  vector_add<<<num_blocks, block_size>>>(A_dev, B_dev, C_dev, array_size);      
  // retruning result to the host vector                                        
  cudaMemcpy(C.data(), C_dev, array_size * sizeof(float),                       
             cudaMemcpyDeviceToHost);                                           
  // releasing the cuda memory objects                                          
  cudaFree(A_dev);                                                              
  cudaFree(B_dev);                                                              
  cudaFree(C_dev);                                                              
  return EXIT_SUCCESS;                                                          
}
