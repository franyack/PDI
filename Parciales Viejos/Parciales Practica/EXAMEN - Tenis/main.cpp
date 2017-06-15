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

void ParcialTenis(string imagen){
	Mat img = imread(imagen);
	namedWindow("original",CV_WINDOW_KEEPRATIO);
	imshow("original",img);
	
	Mat gradiente,transformada;
	vector <vector <Point> > pt;
	HoughComun(img,gradiente,transformada,600,pt);
	namedWindow("tran",CV_WINDOW_KEEPRATIO);
	imshow("tran",transformada);
	waitKey(0);
	
	Point max;
	max.x=-1;
	max.y=-1;
	for(int i=0;i<pt.size();i++) { 
		vector <Point> aux;
		aux=pt[i];
		/*cout<<"LINEA "<<i*/;
		Point p1=aux[0];
//		Point p2=aux[1];
//		cout<<" PUNTO 1: X: "<<p1.x<<" Y: "<<p1.y<<" - PUNTO 2: X: "<<p2.x<<" Y: "<<p2.y<<endl;
		if(p1.y > max.y){
			max.y=p1.y;
			
		}
	}
//	cout<<"max filas "<<max.y;
	
	Mat roi=img(Rect(0,max.y,img.cols,60));
	Mat roi2=img(Rect(0,max.y,img.cols,60));
	vector<Mat> roi_hsv;
	cvtColor(roi,roi,CV_BGR2GRAY);
//	namedWindow("roi",CV_WINDOW_KEEPRATIO);
//	imshow("roi",roi);
	float media = obtenerMedia(roi);
	float desvio = obtenerDesvio(roi);
	for(int i=0;i<roi.rows;i++) { 
		for(int j=0;j<roi.cols;j++) { 
			if((int)roi.at<uchar>(i,j)>media-desvio && (int)roi.at<uchar>(i,j)<media+desvio){
				roi2.at<Vec3b>(i,j)[0]=0;
				roi2.at<Vec3b>(i,j)[1]=0;
				roi2.at<Vec3b>(i,j)[2]=255;
			}
		}
	}
	namedWindow("final",CV_WINDOW_KEEPRATIO);
	imshow("final",img);
	
	

}


int main(int argc, char** argv) {
	
	ParcialTenis("2.jpg");

	waitKey(0);
	return 0;
} 
