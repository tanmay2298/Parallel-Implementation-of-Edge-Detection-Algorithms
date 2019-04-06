import numpy as np
import cv2

def display_img( img , imgtxt ) : 
	cv2.imshow(imgtxt , img )
	cv2.waitKey() 
	cv2.destroyAllWindows() 

def padding( image , rows , cols ) : 
	image_mod = np.random.rand(rows , cols) 
	for i in range(rows) : 
		for j in range(cols) : 
			if i == 0 or i == rows - 1 or j == cols - 1 or j == 0 : 
				image_mod[i][j] = 0 
			else : 
				image_mod[i][j] = image[i-1][j-1]

	image_mod = image_mod.astype(np.uint8)
	print(np.shape(image_mod))
	return image_mod

def get_neighbours(image_mod , i , j) : 
	neighbours = []
	neighbours.append(list(image_mod[i-1][j-1:j+2]))
	neighbours.append( list(image_mod[i][j-1:j+2]) )
	neighbours.append(list(image_mod[i+1][j-1:j+2]))	
	return np.array(neighbours).flatten()


file_path = input( " Enter the image path ")
img_obj = cv2.imread(file_path)
image = cv2.cvtColor(img_obj , cv2.COLOR_BGR2GRAY)
rows , cols = np.shape(image)
local_means = np.random.rand( rows , cols )
local_variance = np.random.rand( rows , cols )


display_img(image , " Original ")

## Step 1 : define Window Size 
window_size = (3,3) 

## Step 2 : pad the input image with 1 layer

image_after_filter = np.random.rand(rows , cols) 
image_mod = padding(image , rows + 1, cols + 1 )

## Step 3 : calculate local means matrix and local variance matrix 

for i in range(rows + 1) : 
	for j in range(cols + 1) : 
		if i == 0 or i == rows or j == cols or j == 0 : 
			continue
		else : 
			# Calculate local mean 
			neighbours = get_neighbours ( image_mod , i , j)
			mean_neigbours = sum(neighbours)/9
			local_means[i-1][j-1] = mean_neigbours

			# Calculate local variance 
			if len(neighbours) < 9 : 
				print( i , j , rows , cols , image_mod[i][j],  neighbours )  
			neighbours = np.reshape( neighbours, (3,3))
			neigbours_square = np.matmul( neighbours , neighbours)
			local_variance[i-1][j-1] = sum(neigbours_square.flatten())/9 - mean_neigbours


## Step 4 : Overall Global Variance for the input image 
print(np.shape(image))
image_sq = np.matmul(image , image.T)
variance = sum(image_sq.flatten()) - sum(image.flatten()) 
variance = variance/((rows)*(cols))

for i in range(rows) : 
	for j in range(cols) : 
		## Step 5 : Set global variance as the minimum variance
		if local_variance[i][j] < variance : 
			local_variance[i][j] = variance 
		## Step 6 : Apply the filter for smoothing
		image_after_filter[i][j] = image[i][j] - (variance/local_variance[i][j])*( image[i][j] - local_means[i][j] )

## Step 7 : Apply Sobel or Prewitt filter for edge [ only horizontal ]
sobel = np.array([1,2,1,0,0,0,-1,-2,-1])
image_for_convolve = padding(image_after_filter , rows + 1 , cols + 1) 

for i in range( rows+ 1 ) : 
	for j in range( cols + 1 ) : 
		if i == 0 or i == rows or j == cols or j == 0 : 
			continue
		else : 
			neighbours = get_neighbours( image_for_convolve , i , j)
			image_after_filter[i - 1][j - 1] = np.dot(neighbours , sobel)

## Step 8 : Detect Edges, based on a threshold

## Step 9 : Display image with edges detected 
display_img(image_after_filter , " Final ")
