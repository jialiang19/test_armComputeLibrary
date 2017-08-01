/*
   This is the c++ program Num has written for me to fill all the weights 
 */  

#define FILE_PATH "/home/odroid/Documents/SummerResearch/ARMLIB_17.06/ComputeLibrary/jprog/CNN/alexnet_data/bvlc_alexnet_explicitly_grouped.binary"
#include <string> 
#include <iostream> 
#include <fstream> 

using namespace std;

class CaffeWeights
{

	private: 
		float *flat_array;
		float ****conv1_w, ****conv2a_w, ****conv2b_w, ****conv3_w,
		      ****conv4a_w, ****conv4b_w, ****conv5a_w, ****conv5b_w;
		float *****conv_all_w[8] = {&conv1_w, &conv2a_w, &conv2b_w, &conv3_w, &conv4a_w, &conv4b_w, &conv5a_w, &conv5b_w};

		float *conv1_b, *conv2a_b, *conv2b_b, *conv3_b, *conv4a_b,
		      *conv4b_b, *conv5a_b, *conv5b_b;
		float **conv_all_b[8] = {&conv1_b, &conv2a_b, &conv2b_b, &conv3_b, &conv4a_b, &conv4b_b, &conv5a_b, &conv5b_b};

		float **fc6_w, **fc7_w, **fc8_w;
		float ***fc_all_w[3] = {&fc6_w, &fc7_w, &fc8_w};

		float *fc6_b, *fc7_b, *fc8_b;
		float **fc_all_b[3] = {&fc6_b, &fc7_b, &fc8_b};

		long array_length;
		int acc;
		int conv1_w_shape[4] = {96, 3, 11, 11};
		int conv2a_w_shape[4] = {128, 48, 5, 5};
		int conv2b_w_shape[4] = {128, 48, 5, 5};
		int conv3_w_shape[4] = {384, 256, 3, 3};
		int conv4a_w_shape[4] = {192, 192, 3, 3};
		int conv4b_w_shape[4] = {192, 192, 3, 3};
		int conv5a_w_shape[4] = {128, 192, 3, 3};
		int conv5b_w_shape[4] = {128, 192, 3, 3};
		int *conv_all_w_shape[8] = {conv1_w_shape, conv2a_w_shape, conv2b_w_shape, conv3_w_shape,
			conv4a_w_shape, conv4b_w_shape, conv5a_w_shape, conv5b_w_shape};
		int fc6_w_shape[2] = {4096, 9216};
		int fc7_w_shape[2] = {4096, 4096};
		int fc8_w_shape[2] = {1000, 4096};
		int *fc_all_w_shape[3] = {fc6_w_shape, fc7_w_shape, fc8_w_shape};


	public:  

		float ****** getWeights() {
				return conv_all_w;  
			    }
		float *** getBias(){
				return conv_all_b; 	
			 }

		float **** getfcWeights() {
				return fc_all_w;
			}
		float *** getfcBias() {
				return fc_all_b;  
			}	
	
		void calculate_weights(){
			const std::string file_path = FILE_PATH;

			std::ifstream binary_file(file_path.c_str(),std::ios::binary);

			array_length = 0;
			// For each layer, calculate size of the weights and bias arrays
			for (int l = 0; l < 8; l++) {
				acc = 1;
				for (int i = 0; i < 4; i++)
					acc *= conv_all_w_shape[l][i];
				array_length += acc + /* bias */conv_all_w_shape[l][0];
			}
			for (int l = 0; l < 3; l++) {
				acc = 1;
				for (int i = 0; i < 2; i++)
					acc *= fc_all_w_shape[l][i];
				array_length += acc + /* bias */fc_all_w_shape[l][0];
			}

			// Read the binary file
			flat_array = new float[array_length];
			binary_file.read(reinterpret_cast<char*>(&flat_array[0]), array_length * sizeof(float));

			// Split the flat array
			float* current_pos = flat_array;
			for (int l = 0; l < 8; l++) {
				alloc(conv_all_w[l], conv_all_w_shape[l]);
				nmap(*conv_all_w[l], conv_all_w_shape[l], current_pos);
				current_pos += conv_all_w_shape[l][0] * conv_all_w_shape[l][1] *
					conv_all_w_shape[l][2] * conv_all_w_shape[l][3];

				nmap(conv_all_b[l], current_pos);
				current_pos += conv_all_w_shape[l][0];
			}
			for (int l = 0; l < 3; l++) {
				alloc(fc_all_w[l], fc_all_w_shape[l]);
				nmap(*fc_all_w[l], fc_all_w_shape[l], current_pos);
				current_pos += fc_all_w_shape[l][0] * fc_all_w_shape[l][1];

				nmap(fc_all_b[l], current_pos);
				current_pos += fc_all_w_shape[l][0];
			}

		}


	protected:
		void alloc(float ***arr2d, int shape[4]) {
			*arr2d = new float*[shape[0]];
		}

		void alloc(float *****arr4d, int shape[4]) {
			// Allocates array of pointers
			*arr4d = new float***[shape[0]];
			for (int i0 = 0; i0 < shape[0]; i0++) {
				(*arr4d)[i0] = new float**[shape[1]];
				for (int i1 = 0; i1 < shape[1]; i1++) {
					(*arr4d)[i0][i1] = new float*[shape[2]];
				}
			}
		}

		void nmap(float **arr_1d, float* arr_flat) {
			// Maps a flat array with pointers
			// (part of arr_flat -> arr_1d)
			*arr_1d = arr_flat;
		}

		void nmap(float **arr_2d, int shape[2], float* arr_flat) {
			// Maps a flat array with pointers
			// (part of arr_flat -> arr_2d)
			for (int i0 = 0; i0 < shape[0]; i0++)
				arr_2d[i0] = &(arr_flat[i0 * shape[1]]);
		}

		void nmap(float ****arr_4d, int shape[4], float* arr_flat) {
			// Maps a flat array with pointers
			// (part of arr_flat -> arr_4d)
			for (int i0 = 0; i0 < shape[0]; i0++)
				for (int i1 = 0; i1 < shape[1]; i1++)
					for (int i2 = 0; i2 < shape[2]; i2++)
						arr_4d[i0][i1][i2] = &(arr_flat[i0 * shape[1] * shape[2] * shape[3] +
								i1 * shape[2] * shape[3] + i2 * shape[3]]);

		}


};  
