#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include "mpi.h"
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

int get_sum(int *arr , int rows , int cols ) {
	int temp_sum = 0 ;
	for (int i = 0 ; i < rows ; i++ )
		for (int j = 0 ; j < cols ; j++ )
		temp_sum = temp_sum + arr[i*cols + j] ;
	return temp_sum ;

}

int main(int argc, char *argv[]){

	int curr_mean , variance , temp , rows , cols , bpp ;
	int near[9] , *near_sq , *image_sq ;
	int sobel[9] = { 1 , 2 , 1, 0 , 0, 0 , -1 , -2 , -1 } ;
	near_sq = (int*)malloc(sizeof(int)*9) ;
	unsigned char *temp_image ;
	temp_image = (unsigned char *)malloc(sizeof(unsigned char)*1000000) ;

	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
//	// 1. Read the image , rows and cols in the image

 	if (rank==0) {
 		temp_image = stbi_load ( "logo.png" , &rows, &cols , &bpp, 1) ;
 /*		for (int i = 60 ; i < 100 ; i++ ){
 			for (int j = 60 ; j < 100 ; j++)
 				printf("%u ", temp_image[i*cols + j]) ;
			printf("\n") ;
 		}
 		printf(" \n  dONE ") ;*/
 	}
 	MPI_Bcast(&rows, 1, MPI_INT,0,MPI_COMM_WORLD);
 	MPI_Bcast(&cols, 1, MPI_INT,0,MPI_COMM_WORLD);
 	MPI_Barrier(MPI_COMM_WORLD) ;


 	int image_filter[rows*cols] ;
 	int image[rows*cols] ;
 	unsigned char image_final[rows*cols] ;
 	unsigned char image_final1[rows*cols] ;
 	unsigned char image_final2[rows*cols] ;

 	int *img_row = (int*)malloc(sizeof(int)*cols) ;
 	int *mean_row = (int*)malloc(sizeof(int)*cols) ;
 	int *var_row = (int*)malloc(sizeof(int)*cols) ;
 	unsigned char *final_char_row = (unsigned char*)malloc(sizeof(unsigned char)*cols) ;
 	unsigned char *char_row = (unsigned char*)malloc(sizeof(unsigned char)*cols) ;
 	unsigned char *char_row1 = (unsigned char*)malloc(sizeof(unsigned char)*cols) ;
 	unsigned char *char_row2 = (unsigned char*)malloc(sizeof(unsigned char)*cols) ;

 	MPI_Scatter(temp_image, cols, MPI_UNSIGNED_CHAR, char_row, cols, MPI_UNSIGNED_CHAR, 0 , MPI_COMM_WORLD) ;
 	MPI_Barrier(MPI_COMM_WORLD) ;

 	for (int j = 0 ; j < cols ; j++ ) {
 		img_row[j] = (int)char_row[j] ;
 		char_row1[j] = (unsigned char)img_row[j] ;
 	}


 	MPI_Gather(img_row, cols, MPI_INT, image, cols , MPI_INT, 0 , MPI_COMM_WORLD) ;
 	MPI_Barrier(MPI_COMM_WORLD) ;
 	MPI_Bcast(image, 1 , MPI_INT , 0 , MPI_COMM_WORLD) ;
 	MPI_Gather(char_row1 , cols , MPI_UNSIGNED_CHAR , image_final1 ,cols , MPI_UNSIGNED_CHAR , 0 , MPI_COMM_WORLD ) ;
 	if (rank == 0 ) {
 		printf(" \n Read and conversion done ") ;
		stbi_write_png("first.png", rows, cols, 1, (const void*)image_final1, rows);
	}

 	if ( rank != 0 && rank != rows - 1) {
		for (int j = 1 ; j < cols - 1 ; j++ ) {
			// 3.1. Compute local mean
			for ( int row_no = 0 ; row_no < 3 ; row_no++ )
				for ( int curr_col = 0 ; curr_col < 3 ; curr_col ++ )
					near[row_no*3 + curr_col ]= image[(rank-1 + row_no)*cols + (j-1) + curr_col] ;
			curr_mean = get_sum(near , 3 , 3)/9 ;
			mean_row[j] = curr_mean ;
			// 3.2. Compute local variance
			near_sq = square_matrix(near , 3 , 3) ;
			var_row[j] = get_sum(near_sq , 3 , 3)/9 - curr_mean ;
		}
		mean_row[0] =  mean_row[1]; mean_row[cols -1] = mean_row[cols - 2] ;
		var_row[0] = var_row[1] ; var_row[cols -1] = var_row[cols - 2] ;
 	}
 	else {
		for (int j = 0 ; j < cols ; j++ ) {
			mean_row[j] = img_row[j] ;
			var_row[j] = img_row[j] ;
		}
 	}


 	if ( rank == 0 ) {
 		printf(" \n mean and variance computed done ") ;
		image_sq = square_matrix(image , rows , cols ) ;
		variance = get_sum(image_sq , 3, 3) - get_sum(image , 3 , 3) ;
		variance = variance/(rows*cols) ;
 	}

 	MPI_Bcast(&variance, 1 , MPI_INT , 0 , MPI_COMM_WORLD) ;
 	MPI_Barrier(MPI_COMM_WORLD) ;


 	for(int j = 0 ; j < cols ; j++ ) {
		if (var_row[j] > variance)
			var_row[j] = variance ;
		img_row[j] = img_row[j] - (variance/var_row[j]) * (img_row[j] - mean_row[j]) ;
	}


 	for (int j = 0 ; j < cols ; j++ ) {
 		char_row2[j] = (unsigned char)img_row[j] ;
 	}
 	MPI_Gather(char_row2 , cols , MPI_UNSIGNED_CHAR , image_final2 ,cols , MPI_UNSIGNED_CHAR , 0 , MPI_COMM_WORLD ) ;
 	MPI_Barrier(MPI_COMM_WORLD) ;
 	if ( rank == 0 ) {
 		printf("\n Image filtered ") ;
 		stbi_write_png("second.png", rows, cols, 1, (const void*)image_final2, rows);
 	}
	// 5. Apply horizontal sobel filter for edge detection

 	MPI_Gather(img_row, cols, MPI_INT, image_filter, cols , MPI_INT, 0 , MPI_COMM_WORLD) ;
 	MPI_Barrier(MPI_COMM_WORLD) ;
 	MPI_Bcast(image_filter, 1 , MPI_INT , 0 , MPI_COMM_WORLD) ;
 	MPI_Barrier(MPI_COMM_WORLD) ;

 	if ( rank > 0 && rank < rows - 1  ) {
		for (int j = 1 ; j < cols - 1; j++ ) {
			temp = 0 ;
			for ( int row_no = 0 ; row_no < 3 ; row_no++ )
				for ( int curr_col = 0 ; curr_col < 3 ; curr_col ++ ) {
					near[row_no*3 + curr_col ]= image_filter[(rank-1 + row_no)*cols + (j-1) + curr_col] ;
					temp = temp + near[row_no*3 + curr_col ]*sobel[row_no*3 + curr_col ] ;
				}
			final_char_row[j] = (unsigned char)temp ;
		}
		final_char_row[0] = final_char_row[1] ;
		final_char_row[cols -1] = final_char_row[cols - 2] ;
	}
 	else {
 		for ( int j = 0 ; j < cols ; j++)
 			final_char_row[j] =  (unsigned char)img_row[j];
 	}
	MPI_Barrier(MPI_COMM_WORLD) ;
	MPI_Gather(final_char_row, cols, MPI_UNSIGNED_CHAR, image_final, cols , MPI_UNSIGNED, 0 , MPI_COMM_WORLD) ;
	// 6. Display the image
	MPI_Barrier(MPI_COMM_WORLD) ;
	if (rank == 0) {
		stbi_write_png("mpi_image.png", rows, cols, 1, (const void*)image_final, rows);
		printf(" Processing complete , open mpi_image.png to see results \n");
	}

	MPI_Finalize();
	}
