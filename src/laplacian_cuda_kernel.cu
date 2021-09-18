#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cublas_v2.h>
#define BLOCK_DIM_1 16
#define BLOCK_DIM_2 256

template <typename scalar_t>
__global__ void compute_squared_norm(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> array,
    int point_num, int dimension, scalar_t* __restrict__ norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex < point_num){
        scalar_t sum = 0;
        for (int i = 0; i < dimension; i++){
            scalar_t val = array[xIndex][i];
            sum += val * val;
        }
        norm[xIndex] = sum;
    }
}

template <typename scalar_t>
__global__ void add_norm(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> array,
    int point_num, scalar_t* __restrict__ norm){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    unsigned int yyIndex = blockIdx.x * blockDim.x + ty;
    __shared__ scalar_t shared_vec[2][BLOCK_DIM_1];
    if (tx == 0 && yIndex < point_num)
        shared_vec[0][ty] = norm[yIndex];
    else if (tx == 1 && yyIndex < point_num)
        shared_vec[1][ty] = norm[yyIndex];
    __syncthreads();
    if (xIndex < point_num && yIndex < point_num){
        array[xIndex][yIndex] = shared_vec[0][ty];
        array[xIndex][yIndex] += shared_vec[1][tx];
    }
}


torch::Tensor laplacian_cuda_forward(torch::Tensor input, int k) {
    cudaSetDevice(input.device().index());
    const int point_num = (int)input.size(0);
    const int dimension = (int)input.size(1);
    auto options = input.options();
    auto norm2 = torch::empty({point_num}, options);
    auto dist = torch::empty({point_num, point_num}, options);
    int n_block1 = point_num / BLOCK_DIM_1;
    if (point_num % BLOCK_DIM_1 != 0)   n_block1 += 1;
    const dim3 threads1(BLOCK_DIM_1, BLOCK_DIM_1);
    const dim3 blocks1(n_block1, n_block1);

    int n_block2 = point_num / BLOCK_DIM_2;
    if (point_num % BLOCK_DIM_2 != 0) n_block2 += 1;
    const dim3 threads2(BLOCK_DIM_2);
    const dim3 blocks2(n_block2);

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "compute_squared_norm", ([&] {
        compute_squared_norm<scalar_t><<<blocks2, threads2>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            point_num, dimension, norm2.data<scalar_t>()
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "add_norm", ([&] {
        add_norm<scalar_t><<<blocks1, threads1>>>(
            dist.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            point_num, norm2.data<scalar_t>()
        );
    }));

    if (input.dtype() == torch::kFloat32){
        float alpha = -2.0;
        float beta = 1.0;
        stat = cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, point_num, 
            point_num, dimension, &alpha,
            input.data<float>(), dimension,
            input.data<float>(), dimension,
            &beta,
            dist.data<float>(), point_num
        );
    }else{
        double alpha = -2.0;
        double beta = 1.0;
        stat = cublasDgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, point_num, 
            point_num, dimension, &alpha,
            input.data<double>(), dimension,
            input.data<double>(), dimension,
            &beta,
            dist.data<double>(), point_num
        );
    }
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS computation failed\n");        
    }
    cublasDestroy(handle);
    auto topkRes = torch::topk(dist, k, 1, false);
    auto coef = std::get<0>(topkRes);
    auto indj = std::get<1>(topkRes);
    auto median = torch::median(coef);
    coef = coef / median;
    coef = torch::exp(-coef);
    auto coef_sum = coef.sum(1,true);
    coef = coef / coef_sum;
    coef = coef.view(-1);
    options = options.dtype(torch::kLong);
    auto indi = torch::arange(point_num, options).unsqueeze_(1).repeat({1, k});
    auto ind1 = torch::stack({indi, indj}).view({2, -1});
    options = options.dtype(torch::kFloat32);
    options = options.layout(torch::kSparse);
    auto W = torch::sparse_coo_tensor(ind1, coef, {point_num, point_num}, options); 
    return W;
}