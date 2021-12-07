#include <torch/extension.h>
#include <cuda.h>

#define BLOCK_DIM 256
#define GRID_DIM 512

template <typename scalar_t>
__device__ scalar_t map_x(scalar_t sourcex, scalar_t sourcey, scalar_t detx, scalar_t dety) {
    return (sourcex * dety - sourcey * detx) / (dety - sourcey);
}

template <typename scalar_t>
__device__ scalar_t map_y(scalar_t sourcex, scalar_t sourcey, scalar_t detx, scalar_t dety) {
    return (sourcey * detx - sourcex * dety) / (detx - sourcex);
}

template <typename scalar_t>
__device__ scalar_t cweight(scalar_t sourcex, scalar_t sourcey, scalar_t detx, scalar_t dety) {
    return (sourcex - detx) * (sourcex - detx) + (sourcey - dety) * (sourcey - dety);
}

template <typename scalar_t>
__global__ void prj_fan_ea(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
    const scalar_t* __restrict__ s2r, const scalar_t* __restrict__ d2r, 
    const scalar_t* __restrict__ binshift) {
    
    __shared__ unsigned int nblocks;
    __shared__ unsigned int idxchannel;
    __shared__ unsigned int idxview;
    nblocks = ceil(*views / gridDim.y);
    idxchannel = blockIdx.x % nblocks;
    idxview = idxchannel * gridDim.y + blockIdx.y;
    if (idxview >= *views)   return;
    idxchannel = blockIdx.x / nblocks;
    __shared__ scalar_t prj[BLOCK_DIM];
    __shared__ scalar_t dPoint[BLOCK_DIM];
    __shared__ scalar_t coef[BLOCK_DIM];
    __shared__ scalar_t dImage;    
    __shared__ scalar_t sourcex;
    __shared__ scalar_t sourcey;
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
    __shared__ double beta0;
    __shared__ double PI;
    __shared__ double ang_error;
    __shared__ double cosval;
    __shared__ double sinval;
    __shared__ unsigned int dIndex0;

    PI = acos(-1.0);
    ang = idxview * *dAng + *Ang0;
    dImage = *dImg;
    ang_error = abs(ang - round(ang / PI) * PI) * 4 / PI;
    cosval = cos(ang);
    sinval = sin(ang);
    sourcex = - sinval * *s2r;
    sourcey = cosval * *s2r;
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    prj[tx] = 0;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        }
        __syncthreads();
        if (tx == 0){
            coef[tx] = dPoint[tx] - dPoint0;
        } else {
            coef[tx] = dPoint[tx] - dPoint[tx-1];
        }
        __syncthreads();
        for (int i = 0; i < ceil(*height / blockDim.x); i++){
            int idxrow = i * blockDim.x + tx;
            if (idxrow < *height) {
                scalar_t i0y = (*height / 2 - idxrow - 0.5) * dImage;
                scalar_t i0x = - *width / 2 * dImage; 
                int idx0col = floor(((i0y - sourcey) / (- sourcey) * 
                    (dPoint0 - sourcex) + sourcex - i0x) / dImage);
                idx0col = max(idx0col, 0);
                i0x += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_x(sourcex, sourcey, i0x, i0y);
                prebound = max(prebound, dPoint0);
                i0x += dImage;
                scalar_t pixbound = map_x(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;                        
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        i0x += dImage;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][idxrow][idxi] / coef[idxd];
                        prebound = pixbound;
                        idxi ++;                        
                        i0x += dImage;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);                        
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][idxrow][idxi] / coef[idxd];
                        prebound = detbound;
                        atomicAdd(prj+idxd, threadprj);
                        threadprj = 0;
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj != 0) atomicAdd(prj+idxd, threadprj);
            }
        }
        __syncthreads();
        dPoint0 = abs(sourcey) / sqrt((dPoint0 - sourcex) * (dPoint0 - sourcex)  + sourcey * sourcey);
        if (dIndex < *dets) {
            dPoint[tx] = abs(sourcey) / sqrt((dPoint[tx] - sourcex) * (dPoint[tx] - sourcex) + sourcey * sourcey);
            __syncthreads();
            if (tx == 0){
                coef[tx] = (dPoint[tx] + dPoint0) / 2;
            } else {
                coef[tx] = (dPoint[tx] + dPoint[tx-1]) / 2;
            }
            __syncthreads();            
            prj[tx] *= dImage;
            prj[tx] /= coef[tx];
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    } else {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        }
        __syncthreads();
        if (tx == 0){
            coef[tx] = dPoint[tx] - dPoint0;
        } else {
            coef[tx] = dPoint[tx] - dPoint[tx-1];
        }
        __syncthreads();
        for (int i = 0; i < ceil(*width / blockDim.x); i++){
            int idxcol = i * blockDim.x + tx;
            if (idxcol < *width) {
                scalar_t i0x = (idxcol - *width / 2 + 0.5) * dImage;
                scalar_t i0y = - *height / 2 * dImage; 
                int idx0row = floor(((i0x - sourcex) / (- sourcex) * 
                    (dPoint0 - sourcey) + sourcey - i0y) / dImage);
                idx0row = max(idx0row, 0);
                i0y += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_y(sourcex, sourcey, i0x, i0y);
                prebound = max(prebound, dPoint0);
                i0y += dImage;
                scalar_t pixbound = map_y(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        i0y += dImage;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] / coef[idxd];
                        prebound = pixbound;
                        idxi ++;
                        i0y += dImage;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] / coef[idxd];
                        prebound = detbound;
                        atomicAdd(prj+idxd, threadprj);
                        threadprj = 0;
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj != 0) atomicAdd(prj+idxd, threadprj);
            }
        }
        __syncthreads();
        dPoint0 = abs(sourcex) / sqrt((dPoint0 - sourcey) * (dPoint0 - sourcey)  + sourcex * sourcex);
        if (dIndex < *dets) {
            dPoint[tx] = abs(sourcex) / sqrt((dPoint[tx] - sourcey) * (dPoint[tx] - sourcey) + sourcex * sourcex);
            __syncthreads();
            if (tx == 0){
                coef[tx] = (dPoint[tx] + dPoint0) / 2;
            } else {
                coef[tx] = (dPoint[tx] + dPoint[tx-1]) / 2;
            }
            __syncthreads();
            prj[tx] *= dImage;
            prj[tx] /= coef[tx];
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    }   
}

template <typename scalar_t>
__global__ void prj_t_fan_ea(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
    const scalar_t* __restrict__ s2r, const scalar_t* __restrict__ d2r, 
    const scalar_t* __restrict__ binshift) {
    
    __shared__ unsigned int nblocks;
    __shared__ unsigned int idxchannel;
    __shared__ unsigned int idxview;
    nblocks = ceil(*views / gridDim.y);
    idxchannel = blockIdx.x % nblocks;
    idxview = idxchannel * gridDim.y + blockIdx.y;
    if (idxview >= *views)   return;
    idxchannel = blockIdx.x / nblocks;
    __shared__ scalar_t prj[BLOCK_DIM];
    __shared__ scalar_t dPoint[BLOCK_DIM];
    __shared__ scalar_t coef[BLOCK_DIM];    
    __shared__ scalar_t dImage;    
    __shared__ scalar_t sourcex;
    __shared__ scalar_t sourcey;
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
    __shared__ double beta0;
    __shared__ double PI;
    __shared__ double ang_error;
    __shared__ double cosval;
    __shared__ double sinval;
    __shared__ unsigned int dIndex0;

    PI = acos(-1.0);
    ang = idxview * *dAng + *Ang0;
    ang_error = abs(ang - round(ang / PI) * PI) * 4 / PI;
    cosval = cos(ang);
    sinval = sin(ang);
    sourcex = - sinval * *s2r;
    sourcey = cosval * *s2r;
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        }
        __syncthreads();
        dImage = abs(sourcey) / sqrt((dPoint0 - sourcex) * (dPoint0 - sourcex)  + sourcey * sourcey);
        if (dIndex < *dets) {
            prj[tx] = abs(sourcey) / sqrt((dPoint[tx] - sourcex) * (dPoint[tx] - sourcex) + sourcey * sourcey);
        } else {
            prj[tx] = 0;
        }
        __syncthreads();
        if (dIndex < *dets) {
            if (tx == 0){
                coef[tx] = (prj[tx] + dImage) / 2;
            } else {
                coef[tx] = (prj[tx] + prj[tx-1]) / 2;
            }
        }
        __syncthreads();
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                prj[tx] = projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex];
            } else {
                prj[tx] = projection[idxchannel][0][idxview][dIndex];
            }
            prj[tx] *= *dImg;
            prj[tx] /= coef[tx];
            if (tx == 0){
                coef[tx] = dPoint[tx] - dPoint0;
            } else {
                coef[tx] = dPoint[tx] - dPoint[tx-1];
            }
            prj[tx] /= coef[tx];
        }   
        __syncthreads();
        dImage = *dImg;
        for (int i = 0; i < ceil(*height / blockDim.x); i++){
            int idxrow = i * blockDim.x + tx;
            if (idxrow < *height) {
                scalar_t i0y = (*height / 2 - idxrow - 0.5) * dImage;
                scalar_t i0x = - *width / 2 * dImage; 
                int idx0col = floor(((i0y - sourcey) / (- sourcey) * 
                    (dPoint0- sourcex) + sourcex - i0x) / dImage);
                idx0col = max(idx0col, 0);
                i0x += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_x(sourcex, sourcey, i0x, i0y);
                prebound = max(prebound, dPoint0);
                i0x += dImage;
                scalar_t pixbound = map_x(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;                        
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        i0x += dImage;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd];
                        prebound = pixbound;
                        atomicAdd(&(image[idxchannel][0][idxrow][idxi]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        i0x += dImage;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd];
                        prebound = detbound;
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj !=0 ) atomicAdd(&(image[idxchannel][0][idxrow][idxi]), threadprj);
            }
        }
    } else {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        }
        __syncthreads();
        dImage = abs(sourcex) / sqrt((dPoint0 - sourcey) * (dPoint0 - sourcey)  + sourcex * sourcex);
        if (dIndex < *dets) {
            prj[tx] = abs(sourcex) / sqrt((dPoint[tx] - sourcey) * (dPoint[tx] - sourcey) + sourcex * sourcex);
        } else {
            prj[tx] = 0;
        }
        __syncthreads();
        if (dIndex < *dets) {
            if (tx == 0){
                coef[tx] = (prj[tx] + dImage) / 2;
            } else {
                coef[tx] = (prj[tx] + prj[tx-1]) / 2;
            }
        }
        __syncthreads();
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                prj[tx] = projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex];
            } else {
                prj[tx] = projection[idxchannel][0][idxview][dIndex];
            }
            prj[tx] *= *dImg;
            prj[tx] /= coef[tx];
            if (tx == 0){
                coef[tx] = dPoint[tx] - dPoint0;
            } else {
                coef[tx] = dPoint[tx] - dPoint[tx-1];
            }
            prj[tx] /= coef[tx];
        }    
        __syncthreads();
        dImage = *dImg;
        for (int i = 0; i < ceil(*width / blockDim.x); i++){
            int idxcol = i * blockDim.x + tx;
            if (idxcol < *width) {
                scalar_t i0x = (idxcol - *width / 2 + 0.5) * dImage;
                scalar_t i0y = - *height / 2 * dImage; 
                int idx0row = floor(((i0x - sourcex) / (- sourcex) * 
                    (dPoint0- sourcey) + sourcey - i0y) / dImage);
                idx0row = max(idx0row, 0);
                i0y += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_y(sourcex, sourcey, i0x, i0y);
                prebound = max(prebound, dPoint0);
                i0y += dImage;
                scalar_t pixbound = map_y(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        i0y += dImage;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd];
                        prebound = pixbound;
                        atomicAdd(&(image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        i0y += dImage;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd];
                        prebound = detbound;                        
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj !=0 ) atomicAdd(&(image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol]), threadprj);
            }
        }
    }   
}

template <typename scalar_t>
__global__ void bprj_t_fan_ea(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
    const scalar_t* __restrict__ s2r, const scalar_t* __restrict__ d2r, 
    const scalar_t* __restrict__ binshift) {
    
    __shared__ unsigned int nblocks;
    __shared__ unsigned int idxchannel;
    __shared__ unsigned int idxview;
    nblocks = ceil(*views / gridDim.y);
    idxchannel = blockIdx.x % nblocks;
    idxview = idxchannel * gridDim.y + blockIdx.y;
    if (idxview >= *views)   return;
    idxchannel = blockIdx.x / nblocks;
    __shared__ scalar_t prj[BLOCK_DIM];
    __shared__ scalar_t dPoint[BLOCK_DIM];
    __shared__ scalar_t coef[BLOCK_DIM];
    __shared__ scalar_t dImage;    
    __shared__ scalar_t sourcex;
    __shared__ scalar_t sourcey;
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
    __shared__ double beta0;
    __shared__ double PI;
    __shared__ double ang_error;
    __shared__ double cosval;
    __shared__ double sinval;
    __shared__ unsigned int dIndex0;

    PI = acos(-1.0);
    ang = idxview * *dAng + *Ang0;
    dImage = *dImg;
    ang_error = abs(ang - round(ang / PI) * PI) * 4 / PI;
    cosval = cos(ang);
    sinval = sin(ang);
    sourcex = - sinval * *s2r;
    sourcey = cosval * *s2r;
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    prj[tx] = 0;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        }
        __syncthreads();
        if (tx == 0){
            coef[tx] = dPoint[tx] - dPoint0;
        } else {
            coef[tx] = dPoint[tx] - dPoint[tx-1];
        }
        __syncthreads();
        for (int i = 0; i < ceil(*height / blockDim.x); i++){
            int idxrow = i * blockDim.x + tx;
            if (idxrow < *height) {
                scalar_t i0y = (*height / 2 - idxrow - 0.5) * dImage;
                scalar_t i0x = - *width / 2 * dImage; 
                int idx0col = floor(((i0y - sourcey) / (- sourcey) * 
                    (dPoint0 - sourcex) + sourcex - i0x) / dImage);
                idx0col = max(idx0col, 0);
                i0x += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_x(sourcex, sourcey, i0x, i0y);
                scalar_t prepixbound = prebound;
                prebound = max(prebound, dPoint0);
                i0x += dImage;
                scalar_t pixbound = map_x(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;                        
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        i0x += dImage;
                        prepixbound = pixbound;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][idxrow][idxi] / (pixbound - prepixbound) / cweight(sourcex, sourcey, i0x - dImage / 2, i0y);
                        prebound = pixbound;
                        idxi ++;                        
                        i0x += dImage;
                        prepixbound = pixbound;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);                        
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][idxrow][idxi] /  (pixbound - prepixbound) / cweight(sourcex, sourcey, i0x - dImage / 2, i0y);
                        prebound = detbound;
                        atomicAdd(prj+idxd, threadprj);
                        threadprj = 0;
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj != 0) atomicAdd(prj+idxd, threadprj);
            }
        }
        __syncthreads();
        dPoint0 = abs(sourcey) / sqrt((dPoint0 - sourcex) * (dPoint0 - sourcex)  + sourcey * sourcey);
        if (dIndex < *dets) {
            dPoint[tx] = abs(sourcey) / sqrt((dPoint[tx] - sourcex) * (dPoint[tx] - sourcex) + sourcey * sourcey);
            __syncthreads();
            if (tx == 0){
                coef[tx] = (dPoint[tx] + dPoint0) / 2;
            } else {
                coef[tx] = (dPoint[tx] + dPoint[tx-1]) / 2;
            }
            __syncthreads();            
            prj[tx] *= dImage;
            prj[tx] /= coef[tx];
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    } else {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        }
        __syncthreads();
        if (tx == 0){
            coef[tx] = dPoint[tx] - dPoint0;
        } else {
            coef[tx] = dPoint[tx] - dPoint[tx-1];
        }
        __syncthreads();
        for (int i = 0; i < ceil(*width / blockDim.x); i++){
            int idxcol = i * blockDim.x + tx;
            if (idxcol < *width) {
                scalar_t i0x = (idxcol - *width / 2 + 0.5) * dImage;
                scalar_t i0y = - *height / 2 * dImage; 
                int idx0row = floor(((i0x - sourcex) / (- sourcex) * 
                    (dPoint0 - sourcey) + sourcey - i0y) / dImage);
                idx0row = max(idx0row, 0);
                i0y += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_y(sourcex, sourcey, i0x, i0y);
                scalar_t prepixbound = prebound;
                prebound = max(prebound, dPoint0);
                i0y += dImage;
                scalar_t pixbound = map_y(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        i0y += dImage;
                        prepixbound = pixbound;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] / (pixbound - prepixbound) / cweight(sourcex, sourcey, i0x, i0y - dImage / 2);
                        prebound = pixbound;
                        idxi ++;
                        i0y += dImage;
                        prepixbound = pixbound;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] / (pixbound - prepixbound) / cweight(sourcex, sourcey, i0x, i0y - dImage / 2);
                        prebound = detbound;
                        atomicAdd(prj+idxd, threadprj);
                        threadprj = 0;
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj != 0) atomicAdd(prj+idxd, threadprj);
            }
        }
        __syncthreads();
        dPoint0 = abs(sourcex) / sqrt((dPoint0 - sourcey) * (dPoint0 - sourcey)  + sourcex * sourcex);
        if (dIndex < *dets) {
            dPoint[tx] = abs(sourcex) / sqrt((dPoint[tx] - sourcey) * (dPoint[tx] - sourcey) + sourcex * sourcex);
            __syncthreads();
            if (tx == 0){
                coef[tx] = (dPoint[tx] + dPoint0) / 2;
            } else {
                coef[tx] = (dPoint[tx] + dPoint[tx-1]) / 2;
            }
            __syncthreads();
            prj[tx] *= dImage;
            prj[tx] /= coef[tx];
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    }   
}

template <typename scalar_t>
__global__ void bprj_fan_ea(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
    const scalar_t* __restrict__ s2r, const scalar_t* __restrict__ d2r, 
    const scalar_t* __restrict__ binshift) {
    
    __shared__ unsigned int nblocks;
    __shared__ unsigned int idxchannel;
    __shared__ unsigned int idxview;
    nblocks = ceil(*views / gridDim.y);
    idxchannel = blockIdx.x % nblocks;
    idxview = idxchannel * gridDim.y + blockIdx.y;
    if (idxview >= *views)   return;
    idxchannel = blockIdx.x / nblocks;
    __shared__ scalar_t prj[BLOCK_DIM];
    __shared__ scalar_t dPoint[BLOCK_DIM];
    __shared__ scalar_t dImage;    
    __shared__ scalar_t sourcex;
    __shared__ scalar_t sourcey;
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
    __shared__ double beta0;
    __shared__ double PI;
    __shared__ double ang_error;
    __shared__ double cosval;
    __shared__ double sinval;
    __shared__ unsigned int dIndex0;

    PI = acos(-1.0);
    ang = idxview * *dAng + *Ang0;
    ang_error = abs(ang - round(ang / PI) * PI) * 4 / PI;
    cosval = cos(ang);
    sinval = sin(ang);
    sourcex = - sinval * *s2r;
    sourcey = cosval * *s2r;
    dImage = *dImg;
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcex + sin(beta0) / cos(beta0) * sourcey;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcex + sin(beta) / cos(beta) * sourcey;
            }
        }
        __syncthreads();        
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                prj[tx] = projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex];
            } else {
                prj[tx] = projection[idxchannel][0][idxview][dIndex];
            }
        } else {
            prj[tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < ceil(*height / blockDim.x); i++){
            int idxrow = i * blockDim.x + tx;
            if (idxrow < *height) {
                scalar_t i0y = (*height / 2 - idxrow - 0.5) * dImage;
                scalar_t i0x = - *width / 2 * dImage; 
                int idx0col = floor(((i0y - sourcey) / (- sourcey) * 
                    (dPoint0- sourcex) + sourcex - i0x) / dImage);
                idx0col = max(idx0col, 0);
                i0x += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_x(sourcex, sourcey, i0x, i0y);
                scalar_t prepixbound = prebound;
                prebound = max(prebound, dPoint0);
                i0x += dImage;
                scalar_t pixbound = map_x(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;                        
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        i0x += dImage;
                        prepixbound = pixbound;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd] / (pixbound - prepixbound);
                        prebound = pixbound;
                        threadprj /= cweight(sourcex, sourcey, i0x - dImage / 2, i0y);
                        atomicAdd(&(image[idxchannel][0][idxrow][idxi]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        i0x += dImage;
                        prepixbound = pixbound;
                        pixbound = map_x(sourcex, sourcey, i0x, i0y);
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd] / (pixbound - prepixbound);
                        prebound = detbound;
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj !=0 ) {
                    threadprj /= cweight(sourcex, sourcey, i0x - dImage / 2, i0y);
                    atomicAdd(&(image[idxchannel][0][idxrow][idxi]), threadprj);
                } 
            }
        }
    } else {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            beta0 = (*dets / 2 - dIndex0) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (*dets / 2 - dIndex - 1) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        } else {
            beta0 = (dIndex0 - *dets / 2) * *dDet + *binshift + ang;
            dPoint0 = sourcey + cos(beta0) / sin(beta0) * sourcex;
            if (dIndex < *dets) {
                double beta = (dIndex + 1 - *dets / 2) * *dDet + *binshift + ang;
                dPoint[tx] = sourcey + cos(beta) / sin(beta) * sourcex;
            }
        }
        __syncthreads();
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                prj[tx] = projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex];
            } else {
                prj[tx] = projection[idxchannel][0][idxview][dIndex];
            }
        } else {
            prj[tx] = 0;
        }        
        __syncthreads();
        
        for (int i = 0; i < ceil(*width / blockDim.x); i++){
            int idxcol = i * blockDim.x + tx;
            if (idxcol < *width) {
                scalar_t i0x = (idxcol - *width / 2 + 0.5) * dImage;
                scalar_t i0y = - *height / 2 * dImage; 
                int idx0row = floor(((i0x - sourcex) / (- sourcex) * 
                    (dPoint0- sourcey) + sourcey - i0y) / dImage);
                idx0row = max(idx0row, 0);
                i0y += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = map_y(sourcex, sourcey, i0x, i0y);
                scalar_t prepixbound = prebound;
                prebound = max(prebound, dPoint0);
                i0y += dImage;
                scalar_t pixbound = map_y(sourcex, sourcey, i0x, i0y);
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        i0y += dImage;
                        prepixbound = pixbound;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd] / (pixbound - prepixbound);
                        prebound = pixbound;
                        threadprj /= cweight(sourcex, sourcey, i0x, i0y - dImage / 2);
                        atomicAdd(&(image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        i0y += dImage;
                        prepixbound = pixbound;
                        pixbound = map_y(sourcex, sourcey, i0x, i0y);
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd] / (pixbound - prepixbound);
                        prebound = detbound;                        
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }
                }
                if (threadprj !=0 ) {
                    threadprj /= cweight(sourcex, sourcey, i0x, i0y - dImage / 2);
                    atomicAdd(&(image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol]), threadprj);
                } 
            }
        }
    }   
}

template <typename scalar_t>
__global__ void rlfilter(scalar_t* __restrict__ filter,
    const scalar_t* __restrict__ dets, const scalar_t* __restrict__ dDet) {
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;    
    __shared__ double PI;
    __shared__ scalar_t d;
    PI = acos(-1.0);
    d = *dDet;
    if (xIndex < (*dets * 2 - 1)) {
        int x = xIndex - *dets + 1;
        if ((abs(x) % 2) == 1) {
            filter[xIndex] = -1 / (PI * PI * x * x * d * d);
        } else if (x == 0) {
            filter[xIndex] = 1 / (4 * d * d);
        } else {
            filter[xIndex] = 0;
        }
    }
}

torch::Tensor prj_fan_ea_cuda(torch::Tensor image, torch::Tensor options) {
    cudaSetDevice(image.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto s2r = options[8];
    auto d2r = options[9];
    auto binshift = options[10];
    const int channels = static_cast<int>(image.size(0));
    auto projection = torch::empty({channels, 1, views.item<int>(), dets.item<int>()}, image.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "fan_beam_equal_angle_projection", ([&] {
        prj_fan_ea<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), s2r.data<scalar_t>(), 
            d2r.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return projection;
}

torch::Tensor prj_t_fan_ea_cuda(torch::Tensor projection, torch::Tensor options) {
    cudaSetDevice(projection.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto s2r = options[8];
    auto d2r = options[9];
    auto binshift = options[10];
    const int channels = static_cast<int>(projection.size(0));
    auto image = torch::zeros({channels, 1, height.item<int>(), width.item<int>()}, projection.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(projection.type(), "fan_beam_equal_angle_backprojection", ([&] {
        prj_t_fan_ea<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), s2r.data<scalar_t>(), 
            d2r.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return image;
}

torch::Tensor bprj_t_fan_ea_cuda(torch::Tensor image, torch::Tensor options) {
    cudaSetDevice(image.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto s2r = options[8];
    auto d2r = options[9];
    auto binshift = options[10];
    const int channels = static_cast<int>(image.size(0));
    auto projection = torch::empty({channels, 1, views.item<int>(), dets.item<int>()}, image.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "fan_beam_equal_angle_fbp_projection", ([&] {
        bprj_t_fan_ea<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), s2r.data<scalar_t>(), 
            d2r.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return projection;
}

torch::Tensor bprj_fan_ea_cuda(torch::Tensor projection, torch::Tensor options) {
    cudaSetDevice(projection.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto s2r = options[8];
    auto d2r = options[9];
    auto binshift = options[10];
    const int channels = static_cast<int>(projection.size(0));
    auto image = torch::zeros({channels, 1, height.item<int>(), width.item<int>()}, projection.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(projection.type(), "fan_beam_equal_angle_fbp_backprojection", ([&] {
        bprj_fan_ea<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), s2r.data<scalar_t>(), 
            d2r.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return image;
}

torch::Tensor fbp_fan_ea_cuda(torch::Tensor projection, torch::Tensor options) {    
    cudaSetDevice(projection.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto s2r = options[8];
    auto d2r = options[9];
    auto binshift = options[10];
    const int channels = static_cast<int>(projection.size(0));
    auto image = torch::zeros({channels, 1, height.item<int>(), width.item<int>()}, projection.options());
    auto filter = torch::empty({1,1,1,dets.item<int>()*2-1}, projection.options());
    auto rectweight = torch::arange((-dets.item<float>()/2+0.5), dets.item<float>()/2, 1, projection.options());
    rectweight = rectweight * dDet;
    rectweight = torch::cos(rectweight);
    rectweight = rectweight * s2r * dDet;
    rectweight = rectweight.view({1, 1, 1, dets.item<int>()});
    rectweight = projection * rectweight;

    int filterdim = ceil((dets.item<float>()*2-1) / BLOCK_DIM);
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);
    
    AT_DISPATCH_FLOATING_TYPES(projection.type(), "ramp_filter", ([&] {
        rlfilter<scalar_t><<<filterdim, BLOCK_DIM>>>(
            filter.data<scalar_t>(), dets.data<scalar_t>(), dDet.data<scalar_t>());
    }));   
    
    auto filtered_projection = torch::conv2d(rectweight, filter, {}, 1, torch::IntArrayRef({0, dets.item<int>()-1}));    

    AT_DISPATCH_FLOATING_TYPES(projection.type(), "fan_beam_equal_angle_fbp_backprojection", ([&] {
        bprj_fan_ea<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            filtered_projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), s2r.data<scalar_t>(), 
            d2r.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    image = image * dAng / 2;
    return image;
}