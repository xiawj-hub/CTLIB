#include <torch/extension.h>
#include <cuda.h>

#define BLOCK_DIM 256
#define GRID_DIM 512

template <typename scalar_t>
__global__ void prj_para(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
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
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
    __shared__ double PI;
    __shared__ double ang_error;
    __shared__ double cosval;
    __shared__ double sinval;
    __shared__ unsigned int dIndex0;
    __shared__ scalar_t dinterval;

    PI = acos(-1.0);
    ang = idxview * *dAng + *Ang0;
    dImage = *dImg;
    ang_error = abs(ang - round(ang / PI) * PI) * 4 / PI;
    cosval = cos(ang);
    sinval = sin(ang);
    dinterval = dImage / *dDet;
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    prj[tx] = 0;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / cosval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / cosval;
            }
        }
        __syncthreads();
        for (int i = 0; i < ceil(*height / blockDim.x); i++){
            int idxrow = i * blockDim.x + tx;
            if (idxrow < *height) {
                scalar_t i0y = (*height / 2 - idxrow - 0.5) * dImage;
                scalar_t i0x = - *width / 2 * dImage;
                scalar_t pixbound = sinval / cosval * i0y + i0x;
                int idx0col = floor((dPoint0 - pixbound) / dImage);
                idx0col = max(idx0col, 0);
                pixbound += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++; 
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][idxrow][idxi] * dinterval;                      
                        prebound = pixbound;
                        idxi ++;
                        pixbound += dImage;                     
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][idxrow][idxi] * dinterval;
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
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    } else {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / sinval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / sinval;
            }
        }
        __syncthreads();
        for (int i = 0; i < ceil(*width / blockDim.x); i++){
            int idxcol = i * blockDim.x + tx;
            if (idxcol < *width) {
                scalar_t i0x = (idxcol - *width / 2 + 0.5) * dImage;
                scalar_t i0y = - *height / 2 * dImage; 
                scalar_t pixbound = i0y + cosval / sinval * i0x;
                int idx0row = floor((dPoint0 - pixbound) / dImage);
                idx0row = max(idx0row, 0);
                pixbound += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] * dinterval;
                        prebound = pixbound;
                        idxi ++;
                        pixbound += dImage;
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] * dinterval;
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
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    }   
}

template <typename scalar_t>
__global__ void prj_t_para(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
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
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
    __shared__ double PI;
    __shared__ double ang_error;
    __shared__ double cosval;
    __shared__ double sinval;
    __shared__ unsigned int dIndex0;
    __shared__ scalar_t dinterval;

    PI = acos(-1.0);
    ang = idxview * *dAng + *Ang0;
    dImage = *dImg;
    ang_error = abs(ang - round(ang / PI) * PI) * 4 / PI;
    cosval = cos(ang);
    sinval = sin(ang);
    dinterval = dImage / *dDet;
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / cosval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / cosval;
            }
        }
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
                scalar_t pixbound = sinval / cosval * i0y + i0x;
                int idx0col = floor((dPoint0 - pixbound) / dImage);
                idx0col = max(idx0col, 0);
                pixbound += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;                        
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd] * dinterval;
                        prebound = pixbound;
                        atomicAdd(&(image[idxchannel][0][idxrow][idxi]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        pixbound += dImage;
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd] * dinterval;
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
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / sinval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / sinval;
            }
        }
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
                scalar_t pixbound = i0y + cosval / sinval * i0x;
                int idx0row = floor((dPoint0 - pixbound) / dImage);
                idx0row = max(idx0row, 0);
                pixbound += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd] * dinterval;
                        prebound = pixbound;
                        atomicAdd(&(image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        pixbound += dImage;
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd] * dinterval;
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
__global__ void bprj_t_para(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
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
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
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
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    prj[tx] = 0;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / cosval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / cosval;
            }
        }
        __syncthreads();
        for (int i = 0; i < ceil(*height / blockDim.x); i++){
            int idxrow = i * blockDim.x + tx;
            if (idxrow < *height) {
                scalar_t i0y = (*height / 2 - idxrow - 0.5) * dImage;
                scalar_t i0x = - *width / 2 * dImage;
                scalar_t pixbound = sinval / cosval * i0y + i0x;
                int idx0col = floor((dPoint0 - pixbound) / dImage);
                idx0col = max(idx0col, 0);
                pixbound += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++; 
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][idxrow][idxi] / dImage;                      
                        prebound = pixbound;
                        idxi ++;
                        pixbound += dImage;                     
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][idxrow][idxi] / dImage;
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
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    } else {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / sinval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / sinval;
            }
        }
        __syncthreads();
        for (int i = 0; i < ceil(*width / blockDim.x); i++){
            int idxcol = i * blockDim.x + tx;
            if (idxcol < *width) {
                scalar_t i0x = (idxcol - *width / 2 + 0.5) * dImage;
                scalar_t i0y = - *height / 2 * dImage; 
                scalar_t pixbound = i0y + cosval / sinval * i0x;
                int idx0row = floor((dPoint0 - pixbound) / dImage);
                idx0row = max(idx0row, 0);
                pixbound += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound < detbound) {
                        threadprj += (pixbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] / dImage;
                        prebound = pixbound;
                        idxi ++;
                        pixbound += dImage;
                    } else {
                        threadprj += (detbound - prebound) * image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol] / dImage;
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
        if (dIndex < *dets) {
            if (ang_error >= 3 && ang_error < 7) {
                projection[idxchannel][0][idxview][static_cast<unsigned int>(*dets)-1-dIndex] = prj[tx];
            } else {
                projection[idxchannel][0][idxview][dIndex] = prj[tx];
            }
        }
    }   
}

template <typename scalar_t>
__global__ void bprj_para(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> image,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> projection,
    const scalar_t* __restrict__ views, const scalar_t* __restrict__ dets,
    const scalar_t* __restrict__ width, const scalar_t* __restrict__ height,
    const scalar_t* __restrict__ dImg, const scalar_t* __restrict__ dDet,
    const scalar_t* __restrict__ Ang0, const scalar_t* __restrict__ dAng, 
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
    __shared__ scalar_t dPoint0;
    __shared__ double ang;
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
    dIndex0 = blockIdx.z * blockDim.x;
    unsigned int tx = threadIdx.x;
    unsigned int dIndex = dIndex0 + tx;
    __syncthreads();
    if (ang_error <= 1) {
        ang_error = (ang - floor(ang / 2 / PI) * 2 * PI) * 4 / PI;
        if (ang_error >= 3 && ang_error < 7) {
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / cosval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / cosval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / cosval;
            }
        }
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
                scalar_t pixbound = sinval / cosval * i0y + i0x;
                int idx0col = floor((dPoint0 - pixbound) / dImage);
                idx0col = max(idx0col, 0);
                pixbound += idx0col * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0col;
                while (idxi < *width && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;                        
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound){
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd] / dImage;
                        prebound = pixbound;
                        atomicAdd(&(image[idxchannel][0][idxrow][idxi]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        pixbound += dImage;
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd] / dImage;
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
            dPoint0 = ((*dets / 2 - dIndex0) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((*dets / 2 - dIndex - 1) * *dDet + *binshift) / sinval;
            }
        } else {
            dPoint0 = ((dIndex0 - *dets / 2) * *dDet + *binshift) / sinval;
            if (dIndex < *dets) {
                dPoint[tx] = ((dIndex + 1 - *dets / 2) * *dDet + *binshift) / sinval;
            }
        }
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
                scalar_t pixbound = i0y + cosval / sinval * i0x;
                int idx0row = floor((dPoint0 - pixbound) / dImage);
                idx0row = max(idx0row, 0);
                pixbound += idx0row * dImage;
                scalar_t threadprj = 0;
                scalar_t prebound = max(pixbound, dPoint0);
                pixbound += dImage;
                scalar_t detbound = dPoint[0];
                int idxd = 0, idxi = idx0row;
                while (idxi < *height && (idxd + dIndex0) < *dets && idxd < blockDim.x) {
                    if (detbound <= prebound) {
                        idxd ++;
                        if (idxd < blockDim.x) detbound = dPoint[idxd];
                    }else if (pixbound <= prebound) {
                        idxi ++;
                        pixbound += dImage;
                    }else if (pixbound <= detbound) {
                        threadprj += (pixbound - prebound) * prj[idxd] / dImage;
                        prebound = pixbound;
                        atomicAdd(&(image[idxchannel][0][static_cast<int>(*height)-1-idxi][idxcol]), threadprj);
                        threadprj = 0;
                        idxi ++;
                        pixbound += dImage;
                    } else {
                        threadprj += (detbound - prebound) * prj[idxd] / dImage;
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

torch::Tensor prj_para_cuda(torch::Tensor image, torch::Tensor options) {
    cudaSetDevice(image.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto binshift = options[8];
    const int channels = static_cast<int>(image.size(0));
    auto projection = torch::empty({channels, 1, views.item<int>(), dets.item<int>()}, image.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "para_projection", ([&] {
        prj_para<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return projection;
}

torch::Tensor prj_t_para_cuda(torch::Tensor projection, torch::Tensor options) {
    cudaSetDevice(projection.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto binshift = options[8];
    const int channels = static_cast<int>(projection.size(0));
    auto image = torch::zeros({channels, 1, height.item<int>(), width.item<int>()}, projection.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(projection.type(), "para_backprojection", ([&] {
        prj_t_para<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return image;
}

torch::Tensor bprj_t_para_cuda(torch::Tensor image, torch::Tensor options) {
    cudaSetDevice(image.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto binshift = options[8];
    const int channels = static_cast<int>(image.size(0));
    auto projection = torch::empty({channels, 1, views.item<int>(), dets.item<int>()}, image.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "para_projection", ([&] {
        bprj_t_para<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return projection;
}

torch::Tensor bprj_para_cuda(torch::Tensor projection, torch::Tensor options) {
    cudaSetDevice(projection.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto binshift = options[8];
    const int channels = static_cast<int>(projection.size(0));
    auto image = torch::zeros({channels, 1, height.item<int>(), width.item<int>()}, projection.options());
    
    int nblocksx = ceil(views.item<float>() / GRID_DIM) * channels;
    int nblocksy = min(views.item<int>(), GRID_DIM);
    int nblocksz = ceil(dets.item<float>() / BLOCK_DIM);
    const dim3 blocks(nblocksx, nblocksy, nblocksz);

    AT_DISPATCH_FLOATING_TYPES(projection.type(), "para_backprojection", ([&] {
        bprj_para<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    return image;
}

torch::Tensor fbp_para_cuda(torch::Tensor projection, torch::Tensor options) {    
    cudaSetDevice(projection.device().index());
    auto views = options[0];
    auto dets = options[1];
    auto width = options[2];
    auto height = options[3];
    auto dImg = options[4];
    auto dDet = options[5];
    auto Ang0 = options[6];
    auto dAng = options[7];
    auto binshift = options[8];
    const int channels = static_cast<int>(projection.size(0));
    auto image = torch::zeros({channels, 1, height.item<int>(), width.item<int>()}, projection.options());
    auto filter = torch::empty({1,1,1,dets.item<int>()*2-1}, projection.options());
    auto rectweight = projection * dDet;

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

    AT_DISPATCH_FLOATING_TYPES(projection.type(), "para_w_backprojection", ([&] {
        bprj_para<scalar_t><<<blocks, BLOCK_DIM>>>(
            image.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            filtered_projection.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            views.data<scalar_t>(), dets.data<scalar_t>(), width.data<scalar_t>(),
            height.data<scalar_t>(), dImg.data<scalar_t>(), dDet.data<scalar_t>(),
            Ang0.data<scalar_t>(), dAng.data<scalar_t>(), binshift.data<scalar_t>()
        );
    }));
    image = image * dAng / 2;
    return image;
}