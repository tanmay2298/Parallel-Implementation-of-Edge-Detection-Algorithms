#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

void image_save(int *original, int rows, int cols, char* name)
{
	unsigned char * image_final = (unsigned char*)calloc(rows*cols, sizeof(char)) ;
	for(int i = 1 ; i < rows ; i++) {
		for (int j = 1 ; j < cols ; j++ ) {
			image_final[(i-1)*cols + j-1] = (unsigned char)original[(i-1)*cols + j-1] ;

		}
	}
	stbi_write_png(name, rows, cols, 1, (const void*)image_final, rows);
}
__global__ void get_global_variance(int *local_variance, int *local_mean, int *image, int *image_filter, int *variance) {
  int cols = blockDim.x ;
  int var = *variance;
  int r = blockIdx.x;
  int c = threadIdx.x;

  if(local_variance[r * cols + c] < var)
    local_variance[r * cols + c] = var;
  image_filter[r * cols + c] = image[r * cols + c] - (var / local_variance[r * cols + c]) * (image[r * cols + c] - local_mean[r * cols + c]);
}
int get_sum2(int *arr, int rows, int cols) {
  int temp_sum = 0;
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      temp_sum += arr[i * cols + j];
  return temp_sum;
}
__global__ void square_matrix2(int *image, int *image_sq )  {
  int row_id = blockIdx.x ;
  int col_id = threadIdx.x ;
  int columns = blockDim.x ;
  int sum = 0 ;
  for(int k = 0; k < columns ; k++)
	  sum = sum + image[row_id*columns + k]*image[col_id*columns + k] ;
  image_sq[row_id *columns + col_id] = sum ;

}
__device__ void square_matrix1(int *mat,int *result ,int rows, int cols) {
  int temp_sum = 0 ;
  for(int i = 0; i < rows; i++) {
	  for(int j = 0; j < cols; j++) {
		  temp_sum = 0 ;
		  for(int k = 0; k < cols; k++)
			  temp_sum = temp_sum + mat[i*cols + k] * mat[j*cols + k] ;
		  result[i*cols + j] = temp_sum ;
	  }
  }
}
__device__ int get_sum(int *arr, int rows, int cols) {
  int temp_sum = 0;
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      temp_sum += arr[i * cols + j];
  return temp_sum;
}
__device__ void get_neighbours(int *image, int *near , int curr_row, int curr_col, int cols) {
  int next = 0;
  for(int i = curr_row - 1; i < curr_row + 2; i++) {
    for(int j = curr_col - 1; j < curr_col + 2; j++) {
      near[next] = image[i * cols + j];
      next++;
    }
  }
}

__global__ void compute_local_mean_variance(int *image_pad, int *local_mean, int *local_variance) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    int columns = blockDim.x ;
    int near_sq[9] ;
    int near[9] ;
    if(r != 0 && c != 0) {
      get_neighbours(image_pad,near,r, c, columns + 1);
      int curr_mean = get_sum(near, 3, 3) / 9;
      local_mean[(r - 1) *columns + (c - 1)] = curr_mean;
      square_matrix1(near, near_sq , 3, 3);
      local_variance[(r - 1) * columns + (c - 1)] = get_sum(near_sq, 3, 3) / 9 - curr_mean;
    }
}
__global__ void image_padding(int *image, int *image_pad) {
  int r = blockIdx.x;
  int c = threadIdx.x;
  int rows = gridDim.x ;
  int cols = blockDim.x ;
  if(r != 0 && c != 0 && r != rows - 1 && c != cols - 1)
    image_pad[r*cols + c] = image[(r - 1)*(cols - 1) + c - 1];
  else
    image_pad[r*cols + c] = 0;
}
__global__ void loadIMG(char *temp_image, int *image) {

  int r = blockIdx.x;
  int c = threadIdx.x;
  int cols = blockDim.x ;
  image[r *cols + c] = (int) temp_image[r *cols + c];

}
__global__ void sobel_horizontal(int *image_final, int *image_pad, int *sobel) {

  int cols = blockDim.x ;
  int rows = gridDim.x ;
  int r = blockIdx.x;
  int c = threadIdx.x;
  int temp = 0;
  int near[9] ;
  if(r > 0 && c > 0 && r < rows - 1 && c < cols - 1 ) {
    get_neighbours(image_pad, near,r, c, cols);
    for(int k = 0; k < 9; k++)
      temp += near[k] * sobel[k];
    image_final[(r - 1)*(cols-1) + (c - 1)] = temp;
  }
}
void err(int checker) {
	cudaError_t errchck = cudaGetLastError() ;
	if (errchck != cudaSuccess )
		 printf(" %d  %s \n" , checker , cudaGetErrorString(errchck ) ) ;
}
int main() {
  int variance, rows, cols, bpp;
  char name[100] ;
  // 1) Read the image
  unsigned char *temp_image = stbi_load("logo.png", &rows, &cols, &bpp, 1);
  int image[rows * cols];

    // Parallel conversion of char image to int image
  int *p_image;
  char *p_temp_image;
  int checkers = 0 ;

  cudaMalloc((void **)&p_image, sizeof(int) * rows * cols);
  cudaMalloc((void **)&p_temp_image, sizeof(char) * rows * cols);
  cudaMemcpy(p_temp_image, temp_image, sizeof(char) * rows * cols, cudaMemcpyHostToDevice);

  loadIMG<<<rows, cols>>>(p_temp_image, p_image);

  // Declarations
  int *image_sq = (int *)malloc(sizeof(int) * rows * cols);
  int sobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  int image_filter[rows * cols];

  // 2) Padding the Image
  int *p_image_pad;
  cudaMalloc((void **)&p_image_pad, sizeof(int) * (rows + 1) * (cols + 1));
  rows += 1;
  cols += 1 ;
  image_padding<<<rows,cols>>>(p_image, p_image_pad);
  err(100) ;
  rows -= 1;
  cols -= 1;
  // 3) Computing Local Mean and Local Variance

  int *p_local_mean, *p_local_variance;
  cudaMalloc((void **)&p_local_mean, sizeof(int)*rows*cols);
  cudaMalloc((void **)&p_local_variance, sizeof(int)*rows*cols);
  compute_local_mean_variance<<<rows, cols>>>(p_image_pad, p_local_mean, p_local_variance);

  // 4) Get Global Variance
  int *p_image_sq;
  cudaMalloc((void **)&p_image_sq, sizeof(int) * rows * cols);
  square_matrix2<<<rows, cols>>>(p_image, p_image_sq);
  cudaMemcpy(image_sq, p_image_sq, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
  cudaFree(p_image_sq);

  // Get Sum2 Function doesn't need to be parallelized
  variance = get_sum2(image_sq , 3, 3) - get_sum2(image , 3 , 3) ;
  variance = variance / (rows * cols);
  int *p_image_filter, *p_variance;

  cudaMalloc((void **)&p_image_filter, sizeof(int) * rows * cols);
  cudaMalloc((void **)&p_variance, sizeof(int));
  cudaMemcpy(p_variance, &variance, sizeof(int), cudaMemcpyHostToDevice);

  get_global_variance<<<rows, cols>>>(p_local_variance, p_local_mean, p_image, p_image_filter, p_variance);
  cudaMemcpy(image_filter, p_image_filter, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  strcpy(name, "noise_removed.png");
  image_save(image_filter, rows, cols, name);
  cudaDeviceSynchronize() ;
  // 5) Apply horizontal sobel filter for edge detection

  rows += 1; /* Investigate this further */
  cols += 1;
  image_padding<<<rows, cols>>>(p_image_filter, p_image_pad);
  cudaDeviceSynchronize() ;
  rows -= 1;
  cols -= 1;

  cudaFree(p_local_variance);
  cudaFree(p_local_mean);
  cudaFree(p_image);

  int image_final[rows*cols] ;
  int *p_image_final;
  int *p_sobel;

  cudaMalloc((void **)&p_image_final, sizeof(int)*rows*cols);
  cudaMalloc((void **)&p_sobel, sizeof(int) * 9);

  cudaMemcpy(p_sobel, sobel, sizeof(int) * 9, cudaMemcpyHostToDevice);
  sobel_horizontal<<<rows+1, cols+1>>>(p_image_final, p_image_pad, p_sobel);
  cudaMemcpy(image_final, p_image_final, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost);
  err(checkers++) ;

  printf("\n\nFunction 5.2 , %d \n\n" , checkers);
  strcpy(name, "final_image.png");
  image_save(image_final, rows, cols, name);
  printf(" Processing complete , open final_image.png to see results \n");

  cudaFree(p_sobel);
  cudaFree(p_image_pad);
  cudaFree(p_image_final);
  cudaFree(p_local_variance);
  cudaFree(p_local_mean);
  cudaFree(p_image);
	return 0 ;
}

