/* 
   Use AlexNet.h Model Object  
 */  
#define ARM_COMPUTE_CL /* So that OpenCL exceptions get caught too */

#include "Globals.h" 
#include "CL/CLAccessor.h"
#include "CL/Helper.h"
#include "TensorLibrary.h"
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

#include "model_objects/AlexNet.h" 
#include "caffe_weights.h" 

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


int main(int argc, const char **argv)
{
	/* program options */
	int iteration; 
	int debug; 
	string line;
	istringstream iss; 

	cout << "Enter the number of the iteration you wish the program to run\n"; 
	getline(cin, line);  	
	iss.clear(); 
	iss.str(line.c_str());
	iss >> iteration; 

	cout << "Enter 1 for debugging, 0 for no debugging\n"; 
	getline(cin, line); 
	iss.clear(); 
	iss.str(line.c_str());
	iss >> debug; 	

	/* 
	   load the weights from caffe 
	   the CaffeWeights object is from the file "caffe_weights.h"
	 */ 
	CaffeWeights caffeweights;
	caffeweights.calculate_weights();
	float ****** conv_all_w = caffeweights.getWeights(); 
	float ***    conv_all_b = caffeweights.getBias(); 
	float ****   fc_all_w   = caffeweights.getfcWeights(); 
	float ***    fc_all_b   = caffeweights.getfcBias(); 

	/* create the AlexNet Object 
	   the AlexNet object is from the in 
	   "~/Documents/SummerResearch/ARMLIB_17.06/final_ComputeLibrary/ComputeLibrary/tests/model_objects/Alexnet.h" 
	 */ 
	test::model_objects::AlexNet <ICLTensor,
		CLTensor,
		CLSubTensor,
		test::cl::CLAccessor,
		CLActivationLayer,
		CLConvolutionLayer,
		CLFullyConnectedLayer,
		CLNormalizationLayer,
		CLPoolingLayer,
		CLSoftmaxLayer> MyAlexNet;

	/* CLScheduler automatically set up the context and select devices
	   for OpenCL, for more control, you can modify CLScheduler yourself*/ 
	CLScheduler::get().default_init();

	/* this is the set up for input as one image, 
	   when I did for the batch size of more than one image, it start to 
	   seg-falut. I check the ordroid memory. It seems that ordroid is 
	   runing out the memory. This need further investigation*/  

	MyAlexNet.init_weights(1,false);  
	cout << "finishing init_weights" << endl; 
	MyAlexNet.build(); 
	cout << "finishing build" << endl; 
	MyAlexNet.allocate();
	cout << "finishing allocate " << endl;  
     //	MyAlexNet.fill_random(); /* everytime when I include the fill_random part of the Alexnet, compliler */ 
	                         /* will complain. I still need to fix this issue */ 	

	/* 
	   fill the weights and biases 
	   fillWeight and fillBias are good example how to store 
	   the information from the vector into Tensor. 
	 */  
	fillWeights(MyAlexNet.w[0].get(),0, conv_all_w); 	
	fillWeights(MyAlexNet.w[1].get(),1, 2, conv_all_w);
	fillWeights(MyAlexNet.w[2].get(),3, conv_all_w); 
	fillWeights(MyAlexNet.w[3].get(),4, 5, conv_all_w); 
	fillWeights(MyAlexNet.w[4].get(),6, 7, conv_all_w); 

	fillfcWeights(MyAlexNet.w[5].get(),0, fc_all_w); 
	fillfcWeights(MyAlexNet.w[6].get(),1, fc_all_w); 
	fillfcWeights(MyAlexNet.w[7].get(),2, fc_all_w); 

	fillBias(MyAlexNet.b[0].get(),0, conv_all_b); 	
	fillBias(MyAlexNet.b[1].get(),1, 2, conv_all_b);
	fillBias(MyAlexNet.b[2].get(),3, conv_all_b); 
	fillBias(MyAlexNet.b[3].get(),4, 5, conv_all_b); 
	fillBias(MyAlexNet.b[4].get(),6, 7, conv_all_b); 
	fillBias(MyAlexNet.b[5].get(),0, fc_all_b); 
	fillBias(MyAlexNet.b[6].get(),1, fc_all_b); 
	fillBias(MyAlexNet.b[7].get(),2, fc_all_b); 


	/* load the srouce image */   
	if(debug) {
		CLImage* src; 
		int counter; 

		ifstream source;                    // build a read-Stream
		source.open("joutput.txt", ios_base::in);  // open data 
		if(source.fail()){
			cerr << "fail to open joutput.txt, unknow data is stored in src tensor." << endl; 
		}	

		float f;
		counter = 0;
		float*** arr3d;

		arr3d = new float**[3];
		for(int i0 = 0; i0 < 3; i0++){
			arr3d[i0] = new float*[277];
			for(int i1 = 0; i1 < 277; i1++){
				arr3d[i0][i1] = new float[227];
			}
		}

		while(getline(source,line)){
			iss.str(line.c_str());
			iss >> f;
			iss.clear();
			arr3d[counter/227/227][counter/227%227][counter%227] = f;
			counter++;
		}

		src = &(MyAlexNet.input);  
		src->map(); 

		Window input_src_window;
		input_src_window.use_tensor_dimensions(src->info());

		Iterator input_src(src, input_src_window);

		counter = 0;
		execute_window_loop(input_src_window, [&](const Coordinates & id)
				{
				*reinterpret_cast<float *>(input_src.ptr()) =  arr3d[counter/227/227][counter/227%227][counter%227];
				counter++;
				},
				input_src);

		CLScheduler::get().sync();
		src->unmap();
	} 

	/* testing the performance the alex_net */ 
	struct timeval start, end;
	vector <double> timeVec; 

	cout << "start to run! " << endl; 
	for(int i = 0; i < iteration; i++) {
		/* run the examples */ 
		gettimeofday(&start, NULL);
		MyAlexNet.run(); 
		gettimeofday(&end, NULL);
		double elapsed_time = end.tv_sec*1000.0 + end.tv_usec/1000.0 - start.tv_sec*1000.0 - start.tv_usec/1000.0;
		timeVec.push_back(elapsed_time); 	
	}

	cout << "finish runing " << endl;
	sort(timeVec.begin(), timeVec.begin()+timeVec.size()); 

	for(int j = 0; j < timeVec.size(); j++){
		if(j%10==0) cout << endl; 	
		cout << timeVec[j] << " "; 
	} 
	cout << endl; 

	cout << "Median time is " << (timeVec[iteration/2]+timeVec[iteration/2+1])/2 << "ms" <<  endl;  
	//////////////////////////////////////////////////////////////////////////////////////////////
	// The following is just used for testing the specific value at each layer //

	string layer_type, layer_name; 
	vector<int> locationVec; 
	int tmpi; 
	map <string, int> layername_map; 	
	map <string, int> weight_map; 
	int weight_index, layer_index; 

	/* map layer_name with layer_index */ 
	layername_map["data"] = 0; 
	layername_map["conv1"] = 1;
	layername_map["norm1"] = 2; 
	layername_map["pool1"] = 3; 
	layername_map["concat2"] = 4; 
	layername_map["norm2"] = 5; 
	layername_map["pool2"] = 6; 
	layername_map["conv3"] = 7; 
	layername_map["concat5"] = 8; 
	layername_map["pool5"] = 9; 
	layername_map["fct6"] = 10; 
	layername_map["fct7"] = 11;
	layername_map["fct8"] = 12;
	layername_map["prob"] = 13; 

	/* map weight_name with weight_index */
	weight_map["conv1"] = 0; 
	weight_map["conv2"] = 1; 
	weight_map["conv3"] = 2; 
	weight_map["conv4"] = 3; 
	weight_map["conv5"] = 4; 
	weight_map["fc6"] = 5; 
	weight_map["fc7"] = 6; 
	weight_map["fc8"] = 7; 

	if(debug){
		cout << "Now you can access specifc value in the Tensor for "
			<< "the debugging purpose! " << endl; 
	}
	while(debug && 1) {
		locationVec.clear(); 
		cout << "Please Enter the type of the Tensor you want to test" << endl;  
		cout << "Enter blobs or params (you can exit the program by command exit)\n"; 
		getline(cin, layer_type); 
		if(*layer_type.rbegin()=='\r'){
			layer_type.erase(layer_type.length()-1); 
		}
		if(layer_type == "exit") break;  
		cout << "type location" << endl; 
		getline(cin, line); 
		iss.clear(); 
		iss.str(line.c_str()); 	
		iss >> layer_name; 
		while(iss >> tmpi) {
			locationVec.push_back(tmpi); 
		}	
		if(layer_type == "blobs"){
			layer_index = layername_map[layer_name]; 	  
			switch(layer_index) {
				case 0: 
					JPrintTensor3D(&(MyAlexNet.input), locationVec);  
					break; 
				case 1: 
					JPrintTensor3D(&(MyAlexNet.act1_out), locationVec);  
					break; 
				case 2:  
					JPrintTensor3D(&(MyAlexNet.norm1_out), locationVec);  
					break; 
				case 3:  
					JPrintTensor3D(&(MyAlexNet.pool1_out), locationVec);  
					break; 
				case 4:  
					JPrintTensor3D(&(MyAlexNet.act2_out), locationVec);  
					break; 
				case 5:  
					JPrintTensor3D(&(MyAlexNet.norm2_out), locationVec);  
					break; 
				case 6:  
					JPrintTensor3D(&(MyAlexNet.pool2_out), locationVec);  
					break; 
				case 7:  
					JPrintTensor3D(&(MyAlexNet.act3_out), locationVec);  
					break;
				case 8:  
					JPrintTensor3D(&(MyAlexNet.act5_out), locationVec);  
					break; 
				case 9:  
					JPrintTensor3D(&(MyAlexNet.pool5_out), locationVec);  
					break; 
				case 10:  
					JPrintTensor1D(&(MyAlexNet.fc6_out), locationVec);  
					break; 
				case 11:  
					JPrintTensor1D(&(MyAlexNet.fc7_out), locationVec);  
					break; 
				case 12:  
					JPrintTensor1D(&(MyAlexNet.fc8_out), locationVec);  
					break; 
				case 13:  
					JPrintTensor1D(&(MyAlexNet.output), locationVec);  
					break; 

				default: 
					cout << "INVALID INPUT! " << endl; 
			}		

		} else if (layer_type == "params"){
			weight_index = weight_map[layer_name]; 	  
			switch(weight_index) {
				case 0: 
					JPrintTensor4D(MyAlexNet.w[0].get(), locationVec);  
					break; 
				case 1: 
					JPrintTensor4D(MyAlexNet.w[1].get(), locationVec);  
					break; 
				case 2:  
					JPrintTensor4D(MyAlexNet.w[2].get(), locationVec);  
					break; 
				case 3:  
					JPrintTensor4D(MyAlexNet.w[3].get(), locationVec);  
					break; 
				case 4:  
					JPrintTensor4D(MyAlexNet.w[4].get(), locationVec);  
					break; 
				case 5:  
					JPrintTensor4D(MyAlexNet.w[5].get(), locationVec);  
					break; 
				case 6:  
					JPrintTensor4D(MyAlexNet.w[6].get(), locationVec);  
					break; 
				case 7:  
					JPrintTensor4D(MyAlexNet.w[7].get(), locationVec);  
					break;

				default: 
					cout << "INVALID INPUT! " << endl; 
			}
		} else {
			cout << "Invalid layer_type: Enter either blobs or params" << endl; 
			cout << "Try to avoid any space" << endl;  
		} 

	}

	MyAlexNet.clear(); 

} 
