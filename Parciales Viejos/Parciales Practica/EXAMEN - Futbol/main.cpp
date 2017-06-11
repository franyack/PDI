#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include "pdi_functions.h"
#include "utils.h"
#include <vector>
#include <opencv2/core/cvdef.h>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iterator>
#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>


using namespace cv;
using namespace pdi;
using namespace std;

void CopaAmerica(){
	Mat img = imread("f3.png");
	imshow("Original",img);
	Mat lines,lines2;
	int cx,cy,cx1,cy1;
	Hough(img,lines,lines2,200,cy1,cx,1,0); //vertical
	cout<<"cy "<<cy<<endl;
	Hough(img,lines,lines2,500,cy,cx1,0,1); //horizontal
	cout<<"cx "<<cx<<endl;
	
	Mat fich = imread("fich.jpg");
	Mat unl = imread("unl.jpg");
	Mat logo = imread("Logo03.png");
	
	cvtColor(img,img,CV_BGR2HSV);
	cvtColor(fich,fich,CV_BGR2HSV);
	cvtColor(unl,unl,CV_BGR2HSV);
	cvtColor(logo,logo,CV_BGR2HSV);
	
	//Armo un roi distinto de cada logo a insertar, sabiendo las coordenadas x, y por la transformada hough y el tamaño de los logos
	Mat roifich=img(Rect(cx-67,cy+1,65,65));
	Mat roiunl=img(Rect(cx+1,cy+1,65,65));
	Mat roilogo=img(Rect(cx-(398/2),img.rows-60,396,58));
	
	vector<Mat> roifich_hsv;
	vector<Mat> roiunl_hsv;
	vector<Mat> roilogo_hsv;
	
	split(roifich,roifich_hsv);
	split(roiunl,roiunl_hsv);
	split(roilogo,roilogo_hsv);
	
	float mediafich = obtenerMedia(roifich_hsv[0]);
	float desviofich = obtenerDesvio(roifich_hsv[0]);
	
	float mediaunl = obtenerMedia(roiunl_hsv[0]);
	float desviounl = obtenerDesvio(roiunl_hsv[0]);
	
	float medialogo = obtenerMedia(roilogo_hsv[0]);
	float desviologo = obtenerDesvio(roilogo_hsv[0]);
	
	
	
	for(int i=0;i<roifich.rows;i++) { 
		for(int j=0;j<roifich.cols;j++) { 
			if((int)roifich_hsv[0].at<uchar>(i,j) > (mediafich-desviofich) && (int)roifich_hsv[0].at<uchar>(i,j) < (mediafich+desviofich)){
				roifich.at<Vec3b>(i,j) = fich.at<Vec3b>(i,j);
			}
		}
	}
	for(int i=0;i<roiunl.rows;i++) { 
		for(int j=0;j<roiunl.cols;j++) { 
			
			if((int)roiunl_hsv[0].at<uchar>(i,j) > (mediaunl-desviounl) && (int)roiunl_hsv[0].at<uchar>(i,j) < (mediaunl+desviounl)){
				roiunl.at<Vec3b>(i,j) = unl.at<Vec3b>(i,j);
			}
		}
	}
	
	for(int i=0;i<roilogo.rows;i++) { 
		for(int j=0;j<roilogo.cols;j++) { 
			
			if((int)roilogo_hsv[0].at<uchar>(i,j) > (medialogo-desviologo) && (int)roilogo_hsv[0].at<uchar>(i,j) < (medialogo+desviologo)){
				roilogo.at<Vec3b>(i,j) = logo.at<Vec3b>(i,j);
			}
		}
	}
	
	cvtColor(img,img,CV_HSV2BGR);
	
	imshow("sadasdasd",img);
	waitKey(0);
}


int main(int argc, char** argv) {
	/*CopaAmerica();*/

	
	waitKey(0);
	return 0;
} 
