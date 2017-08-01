/*      
	Jhelper.h 
	Jia Liang 
	17 July 2017 
	Purpose: a list of the helper functions that are used to put the weights into
		 CLTensors of the Compute Library, and prints out the information 
		 inside of the CLTensors 
*/ 

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "tests/CL/CLAccessor.h"

#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <algorithm>

using namespace arm_compute; 
using namespace std; 

void fillWeights(CLTensor* input, int pointer, float ******conv_all_w){
        int input_x = input->info()->dimension(0);
        int input_y = input->info()->dimension(1);
        int input_z = input->info()->dimension(2);
        int counter = 0;

        float **** conv_w = *(conv_all_w[pointer]);
        int convw_volume = input_x * input_y * input_z;
        int convw_area= input_x * input_y;
        int convw_width = input_x;
        input->map();

        Window input_window;
        input_window.use_tensor_dimensions(input->info());
        Iterator input_I(input, input_window);

        execute_window_loop(input_window, [&](const Coordinates & id)
                        {

               *reinterpret_cast<float *>(input_I.ptr()) = conv_w[counter/convw_volume][(counter%convw_volume)/convw_area][(counter%convw_area)/convw_width][(counter%convw_width)];
                        counter++;
                        },
                        input_I);
        input->unmap();
}



void fillWeights(CLTensor* input, int pointer, int pointer2, float ******conv_all_w){
        int input_x = input->info()->dimension(0);
        int input_y = input->info()->dimension(1);
        int input_z = input->info()->dimension(2);
        int input_batch = input->info()->dimension(3);
        int counter = 0;

        float **** conv_w = *(conv_all_w[pointer]);
        float **** conv_w2 = *(conv_all_w[pointer2]);

        int convw_volume = input_x * input_y * input_z;
        int convw_area= input_x * input_y;
        int convw_width = input_x;
        int half_point = input_x * input_y * input_z * input_batch / 2;

        input->map();

        Window input_window;
        input_window.use_tensor_dimensions(input->info());
        Iterator input_I(input, input_window);

        execute_window_loop(input_window, [&](const Coordinates & id)
                        {
                if(counter < half_point)
               *reinterpret_cast<float *>(input_I.ptr()) = conv_w[counter/convw_volume][(counter%convw_volume)/convw_area][(counter%convw_area)/convw_width][(counter%convw_width)];
                else
               *reinterpret_cast<float *>(input_I.ptr()) = conv_w2[(counter-half_point)/convw_volume][((counter-half_point)%convw_volume)/convw_area][((counter-half_point)%convw_area)/convw_width][((counter-half_point)%convw_width)];
                counter++;
                        },
                        input_I);
        input->unmap();
}


void fillBias(CLTensor *input, int pointer, float ***conv_all_b){

        int counter = 0;

        float *conv_b = *(conv_all_b[pointer]);
        input->map();
        Window input_window;
        input_window.use_tensor_dimensions(input->info());
        Iterator input_I(input, input_window);

        execute_window_loop(input_window, [&](const Coordinates & id)
                        {

                        *reinterpret_cast<float *>(input_I.ptr()) = conv_b[counter];
                        counter++;
                        },
                        input_I);
        input->unmap();
}



void fillBias(CLTensor *input, int pointer, int pointer2, float ***conv_all_b){


        int counter = 0;
        int half_point = input->info()->dimension(0)/2;
        float *conv_b = *(conv_all_b[pointer]);
        float *conv_b2 = *(conv_all_b[pointer2]);

        input->map();
        Window input_window;
        input_window.use_tensor_dimensions(input->info());
        Iterator input_I(input, input_window);

        execute_window_loop(input_window, [&](const Coordinates & id)
                        {

                        if(counter< half_point)
                        *reinterpret_cast<float *>(input_I.ptr()) = conv_b[counter];
                        else
                        *reinterpret_cast<float *>(input_I.ptr()) = conv_b2[(counter-half_point)];
                        counter++;
                        },
                        input_I);

        input->unmap();
}


void fillfcWeights(CLTensor *input, int pointer, float **** fc_all_w){

        int input_x = input->info()->dimension(0);
        int counter = 0;
        float ** fc_w = *(fc_all_w[pointer]);
        int fcw_width = input_x;

        input->map();

        Window input_window;
        input_window.use_tensor_dimensions(input->info());
        Iterator input_I(input, input_window);

        execute_window_loop(input_window, [&](const Coordinates & id)
                        {

                        *reinterpret_cast<float *>(input_I.ptr()) =  fc_w[counter/fcw_width][counter%fcw_width];
                 counter++;
                        },
                        input_I);

        input->unmap();
}

void JPrintTensor1D(CLTensor* input, vector<int> locations){

        int input_x = input->info()->dimension(0);

        cout << "input_x: " << input_x  << endl;

        float *result_data = new float[input_x];
        input->map();

        Window output_window;
        output_window.use_tensor_dimensions(input->info(),Window::DimY);
        Iterator output_it(input, output_window);

        execute_window_loop(output_window, [&](const Coordinates & id)
        {
        memcpy(result_data, output_it.ptr(), input_x * sizeof(float));
        },
        output_it);

        for(int x = 0; x < input_x; x++){
            cout << result_data[x] << " ";
            if(x%5==4) cout << endl;
        }
        cout << endl;
        input->unmap();
        delete result_data;

}

void JPrintTensor2D(CLTensor* input, vector<int> locations){

        int input_x = input->info()->dimension(0);
        int input_y = input->info()->dimension(1);

        cout << "input_x: " << input_x << " input_y: " << input_y  << endl;

        float *result_data = new float[input_x * input_y];
        input->map();

        Window output_window;
        output_window.use_tensor_dimensions(input->info(),Window::DimY);
        Iterator output_it(input, output_window);

        execute_window_loop(output_window, [&](const Coordinates & id)
        {
        memcpy(result_data + id.y() * input_x, output_it.ptr(), input_x * sizeof(float));
        },
        output_it);

        for(int x = 0; x < input_x; x++){
            cout << result_data[locations[0]*input_x + x] << endl;
            if(x%5==4) cout << endl;
        }
        cout << endl;
        input->unmap();
        delete result_data;

}


void JPrintTensor3D(CLTensor* input, vector<int> locations){

        int input_x = input->info()->dimension(0);
        int input_y = input->info()->dimension(1);
        int input_z = input->info()->dimension(2);

        cout << "input_x: " << input_x << " input_y: " << input_y << " input_z " << input_z << endl;

        for(int i = 0; i < locations.size(); i++) cout << locations[i] << " ";
                                                  cout << endl;

        float *result_data = new float[input_x * input_y * input_z];
        input->map();

        Window output_window;
        output_window.use_tensor_dimensions(input->info(),Window::DimY);
        Iterator output_it(input, output_window);

        execute_window_loop(output_window, [&](const Coordinates & id)
        {
        memcpy(result_data + id.z() * input_y* input_x + id.y() * input_x, output_it.ptr(), input_x * sizeof(float));
        },
        output_it);

        for(int x = 0; x < input_x; x++){
            cout << result_data[locations[0]*input_x*input_y + locations[1]*input_x + x] << " ";
            if(x%5==4) cout << endl;
        }
        cout << endl;
        input->unmap();
        delete result_data;

}


void JPrintTensor4D(CLTensor* input, vector<int> locations){

        int input_x = input->info()->dimension(0);
        int input_y = input->info()->dimension(1);
        int input_z = input->info()->dimension(2);
        int input_batch = input->info()->dimension(3);

        cout << "input_x: " << input_x << " input_y: " << input_y << " input_z " << input_z << "input_batch " << input_batch << endl;
        float *result_data = new float[input_x * input_y * input_z * input_batch];
        input->map();

        Window output_window;
        output_window.use_tensor_dimensions(input->info(),Window::DimY);
        Iterator output_it(input, output_window);

        execute_window_loop(output_window, [&](const Coordinates & id)
        {
        memcpy(result_data + id[3] * input_z * input_y * input_x + id.z() * input_y * input_x  + id.y() * input_x, output_it.ptr(), input_x * sizeof(float));
        },
        output_it);

        for(int x = 0; x < input_x; x++){
            cout << result_data[locations[0]*input_x*input_y*input_z + locations[1]*input_x*input_y +locations[2]*input_x+x] << " ";
              if(x%5==4) cout << endl;
        }
        cout << endl;

}

 
