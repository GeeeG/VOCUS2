/*****************************************************************************
*
* main.cpp file for the saliency program VOCUS2. 
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
* Please cite this paper if you use our method.
*
* Implementation:	  Thomas Werner   (wernert@cs.uni-bonn.de)
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
*
* Version 1.1
*
* This code is published under the MIT License 
* (see file LICENSE.txt for details)
*
******************************************************************************/


#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>

#include "VOCUS2.h"

using namespace std;
using namespace cv;

struct stat sb;


int WEBCAM_MODE = -1;
bool VIDEO_MODE = false;
float MSR_THRESH = 0.75; // most salient region
bool SHOW_OUTPUT = true;
string WRITE_OUT = "";
string WRITE_PATH = "";
string OUTOUT = "";
bool WRITEPATHSET = false;
bool CENTER_BIAS = false;

float SIGMA, K;
int MIN_SIZE, METHOD;

vector<string> split_string(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void print_usage(char* argv[]){
	cout << "\nUsage: " << argv[0] << " [OPTIONS] <input-file(s)>" << endl << endl;

	cout << "===== SALIENCY =====" << endl << endl;

	cout << "   -x <name>" << "\t\t" << "Config file (is loaded first, additional options have higher priority)" << endl << endl;

	cout << "   -C <value>" << "\t\t" << "Used colorspace [default: 1]:" << endl;
	cout << "\t\t   " << "0: LAB" << endl;
	cout << "\t\t   " << "1: Opponent (CoDi)" << endl;
	cout << "\t\t   " << "2: Opponent (Equal domains)\n" << endl;

	cout << "   -f <value>" << "\t\t" << "Fusing operation (Feature maps) [default: 0]:" << endl;
	cout << "   -F <value>" << "\t\t" << "Fusing operation (Conspicuity/Saliency maps) [default: 0]:" << endl;

	cout << "\t\t   " << "0: Arithmetic mean" << endl;
	cout << "\t\t   " << "1: Max" << endl;
	cout << "\t\t   " << "2: Uniqueness weight\n" << endl;

	cout << "   -p <value>" << "\t\t" << "Pyramidal structure [default: 2]:" << endl;

	cout << "\t\t   " << "0: Two independant pyramids (Classic)" << endl;
	cout << "\t\t   " << "1: Two pyramids derived from a base pyramid (CoDi-like)" << endl;
	cout << "\t\t   " << "2: Surround pyramid derived from center pyramid (New)\n" << endl;

	cout << "   -l <value>" << "\t\t" << "Start layer (included) [default: 0]" << endl << endl;

	cout << "   -L <value>" << "\t\t" << "Stop layer (included) [default: 4]" << endl << endl;

	cout << "   -S <value>" << "\t\t" << "No. of scales [default: 2]" << endl << endl;

	cout << "   -c <value>" << "\t\t" << "Center sigma [default: 2]" << endl << endl;

	cout << "   -s <value>" << "\t\t" << "Surround sigma [default: 10]" << endl << endl;
	
	//cout << "   -r" << "\t\t" << "Use orientation [default: off]  " << endl << endl;

	cout << "   -e" << "\t\t" << "Use Combined Feature [default: off]" << endl << endl;


	cout << "===== MISC (NOT INCLUDED IN A CONFIG FILE) =====" << endl << endl;

	cout << "   -v <id>" << "\t\t" << "Webcam source" << endl << endl;

	cout << "   -V" << "\t\t" << "Video files" << endl << endl;

	cout << "   -t <value>" << "\t\t" << "MSR threshold (percentage of fixation) [default: 0.75]" << endl << endl;

	cout << "   -N" << "\t\t" << "No visualization" << endl << endl;
	
	cout << "   -o <path>" << "\t\t" << "WRITE results to specified path [default: <input_path>/saliency/*]" << endl << endl;

	cout << "   -w <path>" << "\t\t" << "WRITE all intermediate maps to an existing folder" << endl << endl;

	cout << "   -b" << "\t\t" << "Add center bias to the saliency map\n" << endl << endl;

}

bool process_arguments(int argc, char* argv[], VOCUS2_Cfg& cfg, vector<char*>& files){
	if(argc == 1) return false;

	int c;

	while((c = getopt(argc, argv, "bnreNhVC:x:f:F:v:l:o:L:S:c:s:t:p:w:G:")) != -1){
		switch(c){
		case 'h': return false;
		case 'C': cfg.c_space = ColorSpace(atoi(optarg)); break;
		case 'f': cfg.fuse_feature = FusionOperation(atoi(optarg)); break;
		case 'F': cfg.fuse_conspicuity = FusionOperation(atoi(optarg)); break;
		case 'v': WEBCAM_MODE = atoi(optarg) ; break;
		case 'l': cfg.start_layer = atoi(optarg); break;
		case 'L': cfg.stop_layer = atoi(optarg); break;
		case 'S': cfg.n_scales = atoi(optarg); break;
		case 'c': cfg.center_sigma = atof(optarg); break;
		case 's': cfg.surround_sigma = atof(optarg); break;
		case 'V': VIDEO_MODE = true; break;
		case 't': MSR_THRESH = atof(optarg); break;
		case 'N': SHOW_OUTPUT = false; break;
		case 'o': WRITEPATHSET = true; WRITE_PATH = string(optarg); break;
		case 'n': cfg.normalize = true; break;
		case 'r': cfg.orientation = true; break;
		case 'e': cfg.combined_features = true; break;
		case 'p': cfg.pyr_struct = PyrStructure(atoi(optarg));
		case 'w': WRITE_OUT = string(optarg); break;
		case 'b': CENTER_BIAS = true; break;
		case 'x': break;
			
		default:
			return false;

		}
	}

	if(MSR_THRESH < 0 || MSR_THRESH > 1){
		cerr << "MSR threshold must be in the range [0,1]" << endl;
		return false;
	}

	if(cfg.start_layer < 0){
		cerr << "Start layer must be positive" << endl;
		return false;
	}

	if(cfg.start_layer > cfg.stop_layer){
		cerr << "Start layer cannot be larger than stop layer" << endl;
		return false;
	}

	if(cfg.n_scales <= 0){
		cerr << "Numbor of scales must be > 0" << endl;
		return false;
	}

	if(cfg.center_sigma <= 0){
		cerr << "Center sigma must be positive" << endl;
		return false;
	}

	if(cfg.surround_sigma <= cfg.center_sigma){
		cerr << "Surround sigma must be positive and largen than center sigma" << endl;
		return false;
	}

	
    for (int i = optind; i < argc; i++)
    	files.push_back(argv[i]);

    if (files.size() == 0 && WEBCAM_MODE < 0) {
    	return false;
    }
	
	if(files.size() == 0 && VIDEO_MODE){
		return false;
	}

	if(WEBCAM_MODE >= 0 && VIDEO_MODE){
		return false;
	}

	return true;
}

// if parameter x specified, load config file
VOCUS2_Cfg create_base_config(int argc, char* argv[]){

	VOCUS2_Cfg cfg;

	int c;

	while((c = getopt(argc, argv, "NbnrehVC:x:f:F:v:l:L:S:o:c:s:t:p:w:G:")) != -1){
		if(c == 'x') cfg.load(optarg);
	}

	optind = 1;

	return cfg;
}


//get most salient region
vector<Point> get_msr(Mat& salmap){
	double ma;
	Point p_ma;
	minMaxLoc(salmap, nullptr, &ma, nullptr, &p_ma);

	vector<Point> msr;
	msr.push_back(p_ma);

	int pos = 0;
	float thresh = MSR_THRESH*ma;

	Mat considered = Mat::zeros(salmap.size(), CV_8U);
	considered.at<uchar>(p_ma) = 1;

	while(pos < (int)msr.size()){
		int r = msr[pos].y;
		int c = msr[pos].x;

		for(int dr = -1; dr <= 1; dr++){
			for(int dc = -1; dc <= 1; dc++){
				if(dc == 0 && dr == 0) continue;
				if(considered.ptr<uchar>(r+dr)[c+dc] != 0) continue;
				if(r+dr < 0 || r+dr >= salmap.rows) continue;
				if(c+dc < 0 || c+dc >= salmap.cols) continue;

				if(salmap.ptr<float>(r+dr)[c+dc] >= thresh){
					msr.push_back(Point(c+dc, r+dr));
					considered.ptr<uchar>(r+dr)[c+dc] = 1;
				}
			}
		}
		pos++;
	}

	return msr;
}

int main(int argc, char* argv[]) {

	VOCUS2_Cfg cfg = create_base_config(argc, argv);
	vector<char*> files; // names of input images or video files

	bool correct = process_arguments(argc, argv, cfg, files);

	if(!correct){
		print_usage(argv);
		return EXIT_FAILURE;
	}

	VOCUS2 vocus2(cfg);

	if(WEBCAM_MODE <= -1 && !VIDEO_MODE){ // if normal image
		double overall_time = 0;
		

		if(stat(WRITE_PATH.c_str(), &sb)!=0){
			if(WRITEPATHSET){ 
				std::cout << "Creating directory..." << std::endl;
				mkdir(WRITE_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			}		
		}

		//test if data is image data
		if(files.size()>=1){
				Mat img = imread(files[0]); 
				if( !(img.channels()==3) ){ 
					std::cout << "ABBORT: Inconsistent Image Data"<< img.channels() << std::endl;
					exit(-1);
				}
		}
		
		//compute saliency
		for(size_t i = 0; i < files.size(); i++){
			cout << "Opening " << files[i] << " (" << i+1 << "/" << files.size() << "), ";
			Mat img = imread(files[i], 1);

			long long start = getTickCount();
			Mat salmap;

			vocus2.process(img);
			
			if(CENTER_BIAS) 
			  salmap = vocus2.add_center_bias(0.00005);
			else 
			  salmap = vocus2.get_salmap();


			long long stop = getTickCount();
			double elapsed_time = (stop-start)/getTickFrequency();
			cout << elapsed_time << "sec" << endl;
			overall_time += elapsed_time;
			
			//WRITE resulting saliency maps to path/directory
			string pathStr(files[i]);
			int found = 0;
			found = pathStr.find_last_of("/\\");
			string path = pathStr.substr(0,found);
			string filename_plus = pathStr.substr(found+1);
			string filename = filename_plus.substr(0,filename_plus.find_last_of("."));
			string WRITE_NAME = filename + "_saliency";
			string WRITE_result_PATH;
			//if no path set, write to /saliency directory
			if(!WRITEPATHSET){
				//get the folder, if possible 
				if(found>=0){
					string WRITE_DIR = "/saliency/";
					if ( !(stat(WRITE_PATH.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))){
						mkdir((path+WRITE_DIR).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
					}
					WRITE_result_PATH = path + WRITE_DIR + WRITE_NAME + ".png";
				}
				else{					//if images are in the source folder 
					string WRITE_DIR = "saliency/";
					if ( !(stat(WRITE_PATH.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))){
						mkdir((WRITE_DIR).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
					}
					WRITE_result_PATH = WRITE_DIR + WRITE_NAME + ".png";
				}
			}
			else{ 
				WRITE_result_PATH = WRITE_PATH+"/"+WRITE_NAME+".png";
			}
			std::cout << "WRITE result to " << WRITE_result_PATH << std::endl << std::endl;
			imwrite(WRITE_result_PATH.c_str(), salmap*255.f);
		
			if(WRITE_OUT.compare("") != 0) vocus2.write_out(WRITE_OUT);
		
			
			vector<Point> msr = get_msr(salmap);

			Point2f center;
			float rad;
			minEnclosingCircle(msr, center, rad);

			if(rad >= 5 && rad <= max(img.cols, img.rows)){
			  	circle(img, center, (int)rad, Scalar(0,0,255), 3);
			}

			if(SHOW_OUTPUT){
				imshow("input", img);
				imshow("saliency normalized", salmap);
				waitKey(0);
			}
			
			img.release();
		}

		cout << "Avg. runtime per image: " << overall_time/(double)files.size() << endl;

	}

	else if(WEBCAM_MODE >= 0 || !VIDEO_MODE){ // data from webcam
		double overall_time = 0;
		int n_frames = 0;

		VideoCapture vc(WEBCAM_MODE);
		if(!vc.isOpened()) return EXIT_FAILURE;

		Mat img;

		while(vc.read(img)){
			n_frames++;
			long start = getTickCount();

			Mat salmap;
			
			vocus2.process(img);

			if(CENTER_BIAS) 
			  salmap = vocus2.add_center_bias(0.00005);
			else 
			  salmap = vocus2.get_salmap();


			long stop = getTickCount();
			double elapsed_time = (stop-start)/getTickFrequency();
			cout << "frame " << n_frames << ": " << elapsed_time << "sec" << endl;
			overall_time += elapsed_time;

			vector<Point> msr = get_msr(salmap);

			Point2f center;
			float rad;
			minEnclosingCircle(msr, center, rad);
			
			if(SHOW_OUTPUT){
				imshow("input (ESC to exit)", img);
				imshow("saliency streched (ESC to exit)", salmap);
				int key_code = waitKey(30);

				if(key_code == 113 || key_code == 27) break;
			}
			img.release();
				
		}
		vc.release();
		cout << "Avg. runtime per frame: " << overall_time/(double)n_frames << endl;
	}
	else{ // Video data
	  for(size_t i = 0; i < files.size(); i++){ // loop over video files
			double overall_time = 0;
			int n_frames = 0;

			VideoCapture vc(files[i]);
		
			if(!vc.isOpened()) return EXIT_FAILURE;

			Mat img;

			while(vc.read(img)){
				n_frames++;
				long start = getTickCount();

				Mat salmap;
			
				vocus2.process(img);

				if(!CENTER_BIAS) salmap = vocus2.get_salmap();
				else salmap = vocus2.add_center_bias(0.5);

				long stop = getTickCount();
				double elapsed_time = (stop-start)/getTickFrequency();
				cout << "frame " << n_frames << ": " << elapsed_time << "sec" << endl;
				overall_time += elapsed_time;

				vector<Point> msr = get_msr(salmap);

				Point2f center;
				float rad;
				minEnclosingCircle(msr, center, rad);


				if(SHOW_OUTPUT){
					imshow("input (ESC to exit)", img);
					imshow("saliency streched (ESC to exit)", salmap);
					int key_code = waitKey(30);

					if(key_code == 113 || key_code == 27) break;
				}

				img.release();
				
			}
			vc.release();
			cout << "Avg. runtime per frame: " << overall_time/(double)n_frames << endl;
		}
	}		

	return EXIT_SUCCESS;
}
