#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<math.h>
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

	int main(int argc, char *argv[]){
	char type[200];
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	double t1 = MPI_Wtime();
	int curr_mean , curr_variance , variance , temp , rows , cols , bpp ;
	unsigned char *temp_image;
	// if(rank == 0)
		temp_image = stbi_load ( "logo.png" , &rows, &cols , &bpp, 1) ;
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	unsigned char *temp_image_arr=(unsigned char *)calloc(cols,sizeof(unsigned char));
	

	// printf("%d %d", rows, cols);
	int *result = (int*)calloc(rows*cols ,sizeof(int)) ;
	int result_arr[cols];
	size = 0;
	int image[rows*cols] ;
	int image_arr[cols];

	int image_new[(rows+1)*(cols+1)];
	int image_new_arr[cols+1];

	int *near , *near_sq , *image_sq ;  
	int sobel[9] = { 1 , 2 , 1, 0 , 0, 0 , -1 , -2 , -1 } ; 
	int local_mean[rows*cols] ; 
	int local_variance[rows*cols] ; 
	int image_filter[rows*cols] ; 

	

//	// 1. Read the image , rows and cols in the image 
	
 // 	strcpy(type, "first_im.png");
	// image_save(image, rows, cols, type);
	/*MPI_Scatter(temp_image,cols,MPI_UNSIGNED_CHAR,temp_image_arr,cols,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
	
	for (int j = 0 ; j < cols ; j++ )
		{
			image_arr[j] = (int)temp_image_arr[j] ;
			printf("%d\t%d\n", image_arr[j], temp_image_arr[j]);
			}
			printf("%d\n", rank);
	// MPI_Barrier(MPI_COMM_WORLD);

	
	MPI_Gather(image_arr,cols,MPI_INT,image,cols,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	strcpy(type, "first_im2.png");
	image_save(image, rows, cols, type);*/

	if(rank < size)
	{
		MPI_Scatter(temp_image,cols,MPI_UNSIGNED_CHAR,temp_image_arr,cols,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
			for (int j = 0 ; j < cols ; j++ )
					image_arr[j] = (int)temp_image_arr[j] ;
			MPI_Barrier(MPI_COMM_WORLD);

			
			MPI_Gather(image_arr,cols,MPI_INT,image,cols,MPI_INT,0,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);

			// if(rank==0){
				stbi_write_png("image1.png", rows, cols, 1, (const void*)image, rows);		
				printf(" Processing complete , open image.png to see results \n");
			// }
			MPI_Barrier(MPI_COMM_WORLD);
// //			2.padding
			MPI_Bcast(image,rows*cols,MPI_INT,0,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank!=0){
			for (int j = 1 ; j < cols+1 ; j++){
				image_new_arr[j] = image[(rank-1)*(cols) + j-1] ;//check with sir, cols -1 initially
			}
		    }
			image_new_arr[0]=0;
			if (rank==0)
			{
				for(int j = 1 ; j < cols+1 ; j++){
					image_new_arr[j]=0;
				}
			}

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Gather(image_new_arr,cols+1,MPI_INT,image,cols+1,MPI_INT,0,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(image,(rows+1)*(cols+1),MPI_INT,0,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
// 			if(rank==0){
// 				stbi_write_png("image2.png", rows+1, cols+1, 1, (const void*)image_pad, rows+1);		
// 				printf(" Processing complete , open image.png to see results \n");
// 			}


//			3.LOCAL MEAN
			
				for(int i = 1 ; i < rows-1 ; i++) {
					for (int j = 1 ; j < cols-1 ; j++ ) {
						// 3.1. Compute local mean 
						near = get_neighbours(image, i , j , cols + 1) ; 
						curr_mean = get_sum(near , 3 , 3)/9 ; 
						local_mean[(i-1)*cols + j-1] = curr_mean ; 
						// 3.2. Compute local variance 
						near_sq = square_matrix(near , 3 , 3) ;
						local_variance[(i-1)*cols + j-1] = get_sum(near_sq , 3 , 3)/9 - curr_mean ; 
					}
				}
			
			MPI_Bcast(local_variance,rows*cols,MPI_INT,0,MPI_COMM_WORLD);
			MPI_Bcast(local_mean,rows*cols,MPI_INT,0,MPI_COMM_WORLD);


			// 4. Get Global variance
			for (int j = 0 ; j < cols ; j++ )  {
				for (int k = 0 ; k < cols ; k++ ) {
					result_arr[j] = result_arr[j] + image[k] * image[j*cols + k] ;				}
			}

			MPI_Gather(result_arr,cols,MPI_INT,result,cols,MPI_INT,0,MPI_COMM_WORLD);
			image_sq = result;

			MPI_Barrier(MPI_COMM_WORLD);

			// if(rank==0){
				int v1= 0 ;
				for (int i = 0 ; i < rows ; i++ )
					for (int j = 0 ; j < cols ; j++ )
						v1 = v1 + image_sq[i*cols + j] ;

				int v2= 0 ;
				for (int i = 0 ; i < rows ; i++ )
					for (int j = 0 ; j < cols ; j++ )
						v2 = v2 + image[i*cols + j] ;
				variance=v1-v2;
				variance = variance/(rows*cols) ;
			// }
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&variance,1,MPI_INT,0,MPI_COMM_WORLD);

			for(int j = 0 ; j < cols ; j++ ) {
				if (local_variance[j] < variance)
					local_variance[j] = variance ;
				image_filter[j] = image[j] - (variance/local_variance[j])*(image[j] - local_mean[j]) ;
			}

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Gather(image_filter,cols,MPI_INT,image_filter,cols,MPI_INT,0,MPI_COMM_WORLD);

			// 5. Apply horizontal sobel filter for edge detection 
			
			//			padding
			MPI_Bcast(image_filter,rows*cols,MPI_INT,0,MPI_COMM_WORLD);
			for (int j = 1 ; j < cols+1 ; j++)
				image_new_arr[j] = image_filter[(rank)*(cols - 1) + j-1] ;//check with sir
			MPI_Gather(image_new_arr,cols+1,MPI_INT,image,cols,MPI_INT,0,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(image,(rows+1)*(cols+1),MPI_INT,0,MPI_COMM_WORLD);

		
			unsigned char * image_final = (unsigned char*)calloc(rows*cols, sizeof(char)) ; 
			for(int i = 1 ; i < rows-3 ; i++) {
				for (int j = 1 ; j < cols-3 ; j++ ) {
					temp = 0 ;
					near = get_neighbours(image_filter , i , j , cols  ) ; 
					for (int k = 0 ; k < 9 ; k++ ) 
						temp = temp + near[k]*sobel[k] ; 
					image_final[(i-1)*cols + j-1] = (unsigned char)temp ; 
		 
				}
			}
			
				



		// 6. Display the image
			
			stbi_write_png("image3.png", rows, cols, 1, (const void*)image_final, rows);
			printf(" Processing complete , open image.png to see results \n");
		
	}
	if (rank==0){


	for(int i = 1 ; i < rows ; i++) {
		for (int j = 1 ; j < cols ; j++ ) {
			image[(i-1)*cols + j-1] = (int)temp_image[(i-1)*cols + j-1] ;

		}
	}
	

	// 2. Pad the image 
	int *image_pad =(int*)calloc((rows+1)*(cols+1), sizeof(int));
	image_pad = image_padding(image , rows + 1, cols + 1) ; 
	
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
	stbi_write_png("image_final.png", rows, cols, 1, (const void*)image_final, rows);		
	printf(" Processing complete , open image.png to see results \n");
	

	double t2 = MPI_Wtime();
	printf("Time taken = %fs\n", (t2 - t1));
	}
	// return 0 ; 

	MPI_Finalize();
	}
