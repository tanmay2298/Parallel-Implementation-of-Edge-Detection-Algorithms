#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

int* image_padding( int *image , int rows , int cols ) {
	// rows and columns are each 1 more than the original image
	int* image_new = (int*)calloc(rows*cols , sizeof(int)) ; 
	for (int i = 1 ; i < rows ; i++ ) {
		for (int j = 1 ; j < cols ; j++ ) {
			image_new[i*cols + j] = image[(i-1)*(cols - 1) + j-1] ;
		}
	}
	return image_new ; 	
}

int* get_neighbours( int *image , int curr_row , int curr_col , int cols ) {
	// Assuming window size of 3*3 
	int *near = (int*)malloc(sizeof(int)*9); 
	int next = 0 ; 
	for(int i = curr_row - 1 ; i < curr_row + 2 ; i++) {
		for(int j = curr_col - 1 ; j < curr_col + 2 ; j++) {
			near[next] = image[i*cols + j] ; 
			next++ ; 
		}
	}
	return near ; 
}

int get_sum(int *arr , int rows , int cols ) {
	int temp_sum = 0 ; 
	for (int i = 0 ; i < rows ; i++ )
		for (int j = 0 ; j < cols ; j++ )
		temp_sum = temp_sum + arr[i*cols + j] ; 
	return temp_sum ; 
	
}

int* square_matrix ( int *mat , int rows , int cols )  {
	int *result = (int*)calloc(rows*cols ,sizeof(int)) ; 
	for (int i = 0 ; i < rows ; i++  ) {
		for (int j = 0 ; j < cols ; j++ )  {
			for (int k = 0 ; k < cols ; k++ ) {
				result[i*cols + j] = result[i*cols + j] + mat[i*cols + k] * mat[j*cols + k] ; 
			}
		}
	}
	return result ; 
}


int main() {
	
	int curr_mean , curr_variance , variance , temp , rows , cols , bpp ; 
	// 1. Read the image , rows and cols in the image 
	unsigned char *temp_image = stbi_load ( "logo.png" , &rows, &cols , &bpp, 1) ; 
	int image[rows*cols] ; 
	//int *image = (int*)malloc(sizeof(int)*rows*cols) ; 
	for (int i = 0 ; i < rows ; i++)
		for (int j = 0 ; j < cols ; j++ ) 
			image[i*cols + j ] = (int)temp_image[i*cols + j] ; 
	// Declarations 
	int *near , *near_sq , *image_sq ;  
	int sobel[9] = { 1 , 2 , 1, 0 , 0, 0 , -1 , -2 , -1 } ; 
	int local_mean[rows*cols] ; 
	int local_variance[rows*cols] ; 
	int image_filter[rows*cols] ; 

	// 2. Pad the image 
	int *image_pad = image_padding(image , rows + 1, cols + 1) ; 
	
	for(int i = 1 ; i < rows ; i++) {
		for (int j = 1 ; j < cols ; j++ ) {
			// 3.1. Compute local mean 
			near = get_neighbours(image_pad , i , j , cols + 1) ; 
			curr_mean = get_sum(near , 3 , 3)/9 ; 
			local_mean[(i-1)*cols + j-1] = curr_mean ; 
			// 3.2. Compute local variance 
			near_sq = square_matrix(near , 3 , 3) ;
			local_variance[(i-1)*cols + j-1] = get_sum(near_sq , 3 , 3)/9 - curr_mean ; 
		}
	}
	free(image_pad) ; 
	// 4. Get Global variance 
	image_sq = square_matrix(image , rows , cols ) ; 
	variance = get_sum(image_sq , 3, 3) - get_sum(image , 3 , 3) ; 
	variance = variance/(rows*cols) ; 

	for(int i = 0 ; i < rows ; i++ ) {
		for(int j = 0 ; j < cols ; j++ ) {
			if (local_variance[i*cols + j] < variance) 
				local_variance[i*cols + j] = variance ; 
			image_filter[i*cols + j] = image[i*cols + j] - (variance/local_variance[i*cols + j])*(image[i*cols + j] - local_mean[i*cols + j]) ; 
		}	
	}	
	// 5. Apply horizontal sobel filter for edge detection 
	image_pad = image_padding(image_filter , rows + 1 , cols + 1 ) ; 

	unsigned char * image_final = (unsigned char*)calloc(rows*cols, sizeof(char)) ; 
	for(int i = 1 ; i < rows ; i++) {
		for (int j = 1 ; j < cols ; j++ ) {
			temp = 0 ;
			near = get_neighbours(image_pad , i , j , cols + 1 ) ; 
			for (int k = 0 ; k < 9 ; k++ ) 
				temp = temp + near[k]*sobel[k] ; 
			image_final[(i-1)*cols + j-1] = (unsigned char)temp ; 
 
		}
	}	
	// 6. Display the image 
	stbi_write_png("image.png", rows, cols, 1, (const void*)image_final, rows);		
	printf(" Processing complete , open image.png to see results \n");
	return 0 ; 
}