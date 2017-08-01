#include <string>
#include <iostream>
#include <fstream>

// #define FILE_PATH   "/home/s1569687/caffe/models/bvlc_alexnet/bvlc_alexnet_explicitly_grouped.binary"

#define FILE_PATH "/home/v1jlian2/home/ARMLIB/jprog/CNN/alexnet_data/bvlc_alexnet_explicitly_grouped.binary"

using namespace std;

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

void map(float **arr_1d, float* arr_flat) {
    // Maps a flat array with pointers
    // (part of arr_flat -> arr_1d)
    *arr_1d = arr_flat;
}

void map(float **arr_2d, int shape[2], float* arr_flat) {
    // Maps a flat array with pointers
    // (part of arr_flat -> arr_2d)
    for (int i0 = 0; i0 < shape[0]; i0++)
        arr_2d[i0] = &(arr_flat[i0 * shape[1]]);
}

void map(float ****arr_4d, int shape[4], float* arr_flat) {
    // Maps a flat array with pointers
    // (part of arr_flat -> arr_4d)
    for (int i0 = 0; i0 < shape[0]; i0++)
        for (int i1 = 0; i1 < shape[1]; i1++)
            for (int i2 = 0; i2 < shape[2]; i2++)
                arr_4d[i0][i1][i2] = &(arr_flat[i0 * shape[1] * shape[2] * shape[3] +
                                                i1 * shape[2] * shape[3] + i2 * shape[3]]);

}


int main() {
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
        map(*conv_all_w[l], conv_all_w_shape[l], current_pos);
        current_pos += conv_all_w_shape[l][0] * conv_all_w_shape[l][1] *
                conv_all_w_shape[l][2] * conv_all_w_shape[l][3];

        map(conv_all_b[l], current_pos);
        current_pos += conv_all_w_shape[l][0];
    }
    for (int l = 0; l < 3; l++) {
        alloc(fc_all_w[l], fc_all_w_shape[l]);
        map(*fc_all_w[l], fc_all_w_shape[l], current_pos);
        current_pos += fc_all_w_shape[l][0] * fc_all_w_shape[l][1];

        map(fc_all_b[l], current_pos);
        current_pos += fc_all_w_shape[l][0];
    }


    return 0;
}
