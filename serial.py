import numpy as np
import cv2

def display_img( img , imgtxt ) : 
	cv2.imshow(imgtxt , img )
	cv2.waitKey() 
	cv2.destroyAllWindows() 

file_path = input( " Enter the image path : ")
img_obj = cv2.imread(file_path)
image = cv2.cvtColor(img_obj , cv2.COLOR_BGR2GRAY)
rows , cols = np.shape(image)
local_means = np.random.rand( rows , cols )
local_variance = np.random.rand( rows , cols )

rows = rows + 1 
cols = cols + 1 
display_img(image)

## Step 1 : define Window Size 
window_size = (3,3) 

## Step 2 : pad the input image with 1 layer
image_mod = np.random.rand(rows , cols) 
for i in range(rows) : 
	for j in range(cols) : 
		if i == 0 or i == rows or j == cols or j == 0 : 
			image_mod[i][j] = 0 
		else : 
			image_mod[i][j] = image[i-1][j-1]

image_mod = image_mod.astype(np.uint8)
display_img(image_mod)

## Step 3 : calculate local means matrix and local variance matrix 

for i in range(rows) : 
	for j in range(cols) : 
		if i == 0 or i == rows or j == cols or j == 0 : 
			continue
		else : 
			# Calculate local mean 
			neighbours = list(image_mod[i-1][j-1:j+2])
			neighbours.append( list(image_mod[i][j-1:j+2]) )
			neighbours.append(list(image_mod[i+1][j-1:j+2]))
			mean_neigbours = sum(neighbours)/9
			local_means[i-1][j-1] = mean_neigbours

			# Calculate local variance 
			neighbours = np.array(neighbours)
			neighbours = np.reshape( neighbours, (3,3))
			neigbours_square = np.matmul( neighbours , neighbours)
			local_variance[i-1][j-1] = sum(neigbours_square.flatten())/9 - mean_neigbours


## Step 4 : Overall Global Variance for the input image 
image_sq = np.matmul(image , image)
variance = sum(image_sq.flatten()) - sum(image.flatten()) 
variance = variance/((rows-1)*(cols-1))

## Step 5 : Clean up the local variance
## Step 6 : Apply the filter for smoothing
## Step 7 : Apply Sobel or Prewitt filter for edge
## Step 8 : Detect Edges, based on a threshold 
## Step 9 : Display image with edges detected 