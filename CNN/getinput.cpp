#include <iostream>
#include <fstream>
#include <sstream> 

using namespace std;

int main(void)

{
    ifstream source;                    // build a read-Stream
    source.open("joutput.txt", ios_base::in);  // open data 
    string line; 
	
    istringstream iss; 
    float f; 
    int counter = 0; 

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
//	cout << f << endl; 	
	iss.clear(); 
	arr3d[counter/227/227][counter/227%227][counter%227] = f; 
	counter++; 	
    }

    cout << arr3d[1][2][3] << endl;   

}



