# [New edge detection algorithms based on adaptive estimation filters](https://www.computer.org/csdl/proceedings-article/acssc/1997/00679191/12OmNqBbHZu)

## Introduction
This is a parallel CUDA implementation in C of the algorithm proposed in the research paper by Woodhall et al.

Woodhall, M. C., and C. S. Lindquist. "New edge detection algorithms based on adaptive estimation filters." Conference Record of the Thirty-First Asilomar Conference on Signals, Systems and Computers (Cat. No. 97CB36136). Vol. 2. IEEE, 1997.

## Understanding the Algorithm
This can be explained step wise :

1) Padding the image
2) Computing local mean of each pixel
3) Computing local variance of each pixel
4) Computing global variance
5) Sobel Filter applied for edge detection

## Results
Tested on an NVIDIA GeForce GT 640 GPU using the NVIDIA Nsight Eclipse Edition IDE

Original

![Screenshot](Results/logo.png)

Sobel Filter Applied (Horizontal)

![Screenshot](Results/image.png)

## Notes
1. serial.py : Serial code in python implemented, code not pythonic in nature. Serves as pseudo code for needs to be done in C.

2. serial.c : Serial code in C implemented, needs a g++ compiler to run. Output stored in image.png 

3. stb_image.h, stb_image_write.h : [public libraries](https://github.com/nothings/stb), used for reading and writing images in C.



