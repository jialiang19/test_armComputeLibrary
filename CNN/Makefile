CC = g++
CFLAGS = -I../.. -I../../include -Iinclude 
arm_alexnet: arm_alexnet.cpp  
	$(CC) $(CFLAGS) arm_alexnet.cpp  -w -std=c++14 -larm_compute -lOpenCL  -L ../../build -o arm_alexnet 
