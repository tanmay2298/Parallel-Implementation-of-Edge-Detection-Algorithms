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

__global__ void sobel_horizontal(unsigned char *image_final, int *image_pad, int *sobel, int *r1, int *c1)
{
  int rows = *r1;
  int cols = *c1;

  int r = blockIdx.x;
  int c = threadIdx.x;

  if(r != 0 && c != 0)
  {
    int temp = 0;
    int *near = get_neighbours(image_pad, r, c, cols + 1);
    for(int k = 0; k < 9; k++)
      temp += near[k] * sobel[k];
    image_final[(r - 1) * cols + (c - 1)] = (unsigned char) temp;
  }
}
__global__ void get_global_variance(int *local_variance, int *local_mean, int *image, int *image_filter, int *variance, int *r1, int *c1)
{
  int rows = *r1;
  int cols = *c1;
  int var = *variance;
  int r = blockIdx.x;
  int c = threadIdx.x;

  if(local_variance[r * cols + c] < var)
    local_variance[r * cols + c] = var;
  image_filter[r * cols + c] = image[r * cols + c] - (var / local_variance[r * cols + c]) * (image[r * cols + c] - local_mean[r * cols + c]);
}
int get_sum2(int *arr, int rows, int cols)
{
  int temp_sum = 0;
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      temp_sum += arr[i * cols + j];
  return temp_sum;
}
__global__ void square_matrix2(int *image, int *image_sq, int *rows, int *cols) // parallelized matrix square fn called for calculating global variance
{
  int id = threadIdx.x;

  for(int i = 0; i < (*cols); i++)
  {
    image_sq[id * (*cols) + i] = 0;
    for(j = 0; j < (*rows); j++)
      image_sq[id * q + i] += image[id * (*rows) + k] * image[k * (*cols) + j];
  }

}
__device__ int* square_matrix1(int *mat, int rows, int cols) // non parallelisable square function called in compute_local_mean_variance
{
  int *result = (int *)calloc(rows * cols, sizeof(int));
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      for(int k = 0; k < cols; k++)
        result[i*cols + j] = result[i*cols + j] + mat[i*cols + k] * mat[j*cols + k] ;
}
__device__ int get_sum(int *arr, int rows, int cols)
{
  int temp_sum = 0;
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      temp_sum += arr[i * cols + j];
  return temp_sum;
}
__device__ int* get_neighbours(int *image, int curr_row, int cols)
{
  int *near = (int *)malloc(sizeof(int) * 9);
  int next = 0;
  for(int i = curr_row - 1; i < curr_row + 2; i++)
  {
    for(int j = curr_col - 1; j < curr_col + 2; j++)
    {
      near[next] = image[i * cols + j];
      next++;
    }
  }
  return near;
}
__global__ void compute_local_mean_variance(int *image_pad, int *local_mean, int *local_variance, int *rows, int *cols)
{
    int r = blockIdx.x;
    int c = threadIdx.x;

    if(r != 0 && c != 0)
    {
      int *near = get_neighbours(image_pad, r, c, (*cols + 1));
      int curr_mean = get_sum(near, 3, 3) / 9;
      local_mean[(r - 1) * (*cols) + (c - 1)] = curr_mean;
      int *near_sq = square_matrix1(near, 3, 3);
      local_variance[(r - 1) * (*cols) + (c - 1)] = get_sum(near_sq, 3, 3) / 9 - curr_mean;
    }
}
__global__ void image_padding(int *image, int *image_pad, int *rows, int *cols)
{
  int r = blockIdx.x;
  int c = threadIdx.x;

  if(r != 0 && c != 0)
    image_pad[r * (*cols) + c] = image[(r - 1) * (*cols - 1) + c - 1];
  else
    image_pad[r * (*cols) + c] = 0;
}
__global__ void loadIMG(char *temp_image, int *image, int *rows, int *cols)
{
  int r = blockIdx.x;
  int c = threadIdx.x;

  image[r * (*cols) + c] = (int) temp_image[r * (*cols) + c];
}
int main()
{
  int curr_mean, curr_variance, variance, temp, rows, cols, bpp;
  // 1) Imread
  unsigned char *temp_image = stbi_load("logo.png", &rows, &cols, &bpp, 1);
  int image[rows * cols];

    // Parallel conversion of char image to int image
  int *p_image;
  char *p_temp_image;
  int *p_row, *p_col;

  cudaMalloc((void **)&p_image, sizeof(int) * rows * cols);
  cudaMalloc((void **)&p_temp_image, sizeof(char) * rows * cols);
  cudaMalloc((void **)&p_row, sizeof(int));
  cudaMalloc((void **)&p_col, sizeof(int));

  cudaMemcpy(p_temp_image, temp_image, sizeof(char) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);

  loadIMG<<<rows, cols>>>(p_temp_image, p_image, p_row, p_col);

  cudaMemcpy(image, p_image, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  // Declarations
  int *near, *near_sq, *image_sq;
  int sobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  int local_mean[rows * cols];
  int local_variance[rows * cols];
  int image_filter[rows * cols];

  // 2) Padding the Image
  int *image_pad;
  int *p_image_pad;

  // Parallelizing image_padding function
  cudaMalloc((void **)&p_image_pad, sizeof(int) * (rows + 1) * (cols + 1));

  rows += 1;
  cols += 1;

  cudaMemcpy(p_image, image, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);

  image_padding<<<(rows + 1), (cols + 1)>>>(p_image, p_image_pad, p_row, p_col);

  cudaMemcpy(image_pad, p_image_pad, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  rows -= 1;
  cols -= 1;

  // 3) Computing Local Mean and Local Variance
  // Note : This is not completely parallelizable due to data dependency

  int *p_local_mean, *p_local_variance;

  cudaMalloc((void **)&p_local_mean, sizeof(int) * (rows) * (cols));
  cudaMalloc((void **)&p_local_mean, sizeof(int) * (rows) * (cols));

  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_image_pad, image_pad, sizeof(int) * (rows + 1) * (cols + 1), cudaMemcpyDeviceToHost);

  compute_local_mean_variance<<<rows, cols>>>(p_image_pad, p_local_mean, p_local_variance, p_row, p_col);

  cudaMemcpy(local_mean, p_local_mean, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
  cudaMemcpy(local_variance, p_local_variance, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  // 4) Get Global Variance
  int *p_image_sq;

  // Square Matrix2 Function
  cudaMalloc((void **)&p_image_sq, sizeof(int) * rows * cols);

  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_image, image, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);

  square_matrix2<<<1, rows>>>(p_image, p_image_sq, p_row, p_col);

  cudaMemcpy(image_sq, p_image_sq, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  // Get Sum2 Function doesn't need to be parallelized
  variance = get_sum2(image_sq , 3, 3) - get_sum2(image , 3 , 3) ;
  variance = variance / (rows * cols);

  int *p_image_filter, *p_variance;

  cudaMalloc((void **)&p_image_filter, sizeof(int) * rows * cols);

  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_variance, &variance, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_image, image, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_local_mean, local_mean, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_local_variance, local_variance, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);

  get_global_variance<<<rows, cols>>>(p_local_variance, p_local_mean, p_image, p_image_filter, p_variance, p_row, p_col);

  cudaMemcpy(image_filter, p_image_filter, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  // 5) Apply horizontal sobel filter for edge detection

  // Image padding
  rows += 1;
  cols += 1;

  cudaMemcpy(p_image_filter, image_filter, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);

  image_padding<<<rows, cols>>>(p_image_filter, p_image_pad, p_row, p_col);

  cudaMemcpy(image_pad, p_image_pad, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

  rows -= 1;
  cols -= 1;

  unsigned char *image_final = (unsigned char *) calloc(rows * cols, sizeof(char));
  unsigned char *p_image_final;

  int *p_sobel;

  cudaMalloc((void **)&p_image_final, sizeof(int) * rows * cols);
  cudaMalloc((void **)&p_sobel, sizeof(int) * 9);

  cudaMemcpy(p_image_pad, image_pad, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_image_final, image_final, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(p_row, &rows, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col, &cols, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_sobel, sobel, sizeof(int) * 9, cudaMemcpyHostToDevice);

  horizontal_sobel<<<rows, cols>>>(p_image_final, p_image_pad, p_sobel, p_row, p_col);

  stbi_write_png("image.png", rows, cols, 1, (const void*)image_final, rows);
	printf(" Processing complete , open image.png to see results \n");
	return 0 ;
}
