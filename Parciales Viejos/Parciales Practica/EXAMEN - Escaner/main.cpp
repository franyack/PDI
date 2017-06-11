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




int main(int argc, char** argv) {
	
	Mat img = imread("10.png");
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	imshow("Original",img);
	Mat img2 = img.clone();
	cvtColor(img,img,CV_BGR2GRAY);
	Mat espectro=espectroFrecuencia(img);
	
	namedWindow("Espectro",CV_WINDOW_KEEPRATIO);
	imshow("Espectro",espectro);
	
	Mat umbral(espectro.size(),espectro.type());
	for(int i=0;i<espectro.rows;i++) { 
		for(int j=0;j<espectro.cols;j++) { 
			if (espectro.at<float>(i,j)> 130/255.0){
				umbral.at<float>(i,j)=1;
			}
			else umbral.at<float>(i,j)=0;
		}
	}
	
	namedWindow("Umbral1",CV_WINDOW_KEEPRATIO);
	imshow("Umbral1",umbral);
	
	umbral = aplicar_filtro_promediador(umbral,5);
	
	for(int i=0;i<umbral.rows;i++) { 
		for(int j=0;j<umbral.cols;j++) { 
			if (umbral.at<float>(i,j)>80/255.0){
				umbral.at<float>(i,j)=1;
			}
			else{
				umbral.at<float>(i,j)=0;
			}
		}
	}
	
	namedWindow("Umbral",CV_WINDOW_KEEPRATIO);
	imshow("Umbral",umbral);
	
	int ii=0;
	int jj=0;
	for(int i=0;i<umbral.rows;i++) { 
		for(int j=0;j<umbral.cols;j++) { 
			if (umbral.at<float>(i,j)==1 && ii==0 && jj==0){
				ii=i;
				jj=j;
			}
			
		}
	}
	
	double angulo;
	
	if (jj<(int)umbral.cols/2){
		double a=abs(umbral.cols/2-jj);
		double b=abs(umbral.rows/2-ii);
		double division=a/b;
		angulo=atan(division)*180/M_PI;
		img2=rotate(img2,-angulo);
	}
	else{
		double a=abs(jj-umbral.cols/2);
		double b=abs(umbral.rows/2-ii);
		double division=a/b;
		angulo=atan(division)*180/M_PI;
		img2=rotate(img2,angulo);
	}
	cout<<"Cantidad filas:"<<espectro.rows/2<<" Cantidad columnas:"<<espectro.cols/2<<endl<<" I:"<<ii<<" J:"<<jj<<" Angulo:"<<angulo<<endl;
	//	info(umbral);
	namedWindow("Umbral",CV_WINDOW_KEEPRATIO);
	imshow("Umbral",umbral);
	
	
	cout<<"Se roto "<<angulo<<endl;
	
	namedWindow("Rotado",CV_WINDOW_KEEPRATIO);
	imshow("Rotado",img2);
	
	
	
	waitKey(0);
	return 0;
} 
