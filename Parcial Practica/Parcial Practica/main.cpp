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

	
	Mat img = imread("1.png");
	imshow("swad",img);
	
////	cvtColor(img,img,CV_BGR2HSV);
//	vector<Mat> img_bgr;
//	split(img,img_bgr);
//	
	Mat roi = img(Rect(162,212,10,10));
	vector<Mat>roi_bgr;
	split(roi,roi_bgr);
	imshow("roi",roi);
	
	float mediaB = obtenerMedia(roi_bgr[0]);
	float mediaG = obtenerMedia(roi_bgr[1]);
	float mediaR = obtenerMedia(roi_bgr[2]);
	
	float desvioB = obtenerDesvio(roi_bgr[0]);
	float desvioG = obtenerDesvio(roi_bgr[1]);
	float desvioR = obtenerDesvio(roi_bgr[2]);
	
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
	
	inRange(img,Scalar(mediaB-desvioB,mediaG-desvioG,mediaR-desvioR),Scalar(mediaB+desvioB,mediaG+desvioG,mediaR+desvioR),mascara);
	
	imshow("Mascara",mascara);
	
//	
//////	Mat lines,lines2;
////	int cx,cy,cx1,cy1;
//////	Hough(mascara,lines,lines2,30,cy1,cx,1,0); //vertical
////	
//////	imshow("line1",lines);
////	
//////	Hough(mascara,lines,lines2,30,cy1,cx,0,1); //horizontal
////	vector<Vec2f> lines;
////	// detect lines
////	HoughLines(mascara, lines, 1, CV_PI/180, 10, 0, 0 );
////	
////	float rho = lines[0][0], theta = lines[0][1];
////	cout<<"rho "<<rho<<endl;
////	cout<<"theta "<<theta<<endl;
//	
	Mat Gradiente2;
	
	Canny(mascara,Gradiente2,50,200,3); //Detecto los bordes de la imagen
//	namedWindow("Original",CV_WINDOW_KEEPRATIO);
//	imshow("Original",img);
//	namedWindow("Bordes",CV_WINDOW_KEEPRATIO);
//	imshow("Bordes",Gradiente2);
	
	
	
	vector <Vec4i> lines2;
	HoughLinesP(Gradiente2,lines2,1,CV_PI/180, 30, 10, 10  ); 
	Mat transformadaP;
	cvtColor(Gradiente2, transformadaP, CV_GRAY2BGR);
	imshow("Gradiente",Gradiente2);
	
	
	
	//Los parametros son los mismos, pero los ultimos dos son:
	//* El numero minimo de puntos que se puede formar una linea, las lineas con menos de estos puntos no se tienen en cuenta
	//* Separacion maxima entre dos puntos a considerar en la misma recta
	for(size_t i=0; i<lines2.size(); i++) { 
		Vec4i  l=lines2[0];
		line(transformadaP ,Point(l[0],l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA); 
	}
	namedWindow("Transformada HoughP",CV_WINDOW_KEEPRATIO);
	imshow("Transformada HoughP",transformadaP);
	
	int minx=9999;
	int miny=9999;
	int maxx=-1;
	int maxy=-1;
	for(int i=0;i<lines2.size();i++) { 
		Vec4i  v=lines2[0];
		if (v[0]<minx){
			minx=v[0];
		}
		if (v[1]<miny){
			miny=v[1];
		}
		if (v[0]>maxx){
			maxx=v[0];
		}
		if (v[1]>maxy){
			maxy=v[1];
		}
		if (v[2]<minx){
			minx=v[2];
		}
		if (v[3]<miny){
			miny=v[3];
		}
		if (v[2]>maxx){
			maxx=v[2];
		}
		if (v[3]>maxy){
			maxy=v[3];
		}
	}
	
//	cout<<"maxx "<<maxx<<endl;
//	cout<<"minx "<<minx<<endl;
//	cout<<"maxy "<<maxy<<endl;
//	cout<<"miny "<<miny<<endl;

	float difx = abs(maxx-minx);
	
	float dify = abs(miny-maxy);
	
	
//	float uno = sqrt(pow(difx,2) + pow(dify,2));
////	cout<<"Uno "<<uno<<endl;
//	
	float angulo = atan(difx/dify)*180/M_PI;
	cout<<"angulo: "<<180+angulo;
////	if(minx-maxx < 0){
////		cout<<180+angulo;
////	}else{
////		cout<<180-angulo;
////	}
//	
//	//Si es la imagen uno: 180 + angulo
////	cout<<180+angulo;
//	//si es la imagen 2: 180 - angulo
//	cout<<180+angulo;
////	imshow("line",lines);

	
	
	waitKey(0);
	return 0;
} 
