#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include "pdi_functions.h"
#include "funciones_FS.h"
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
#include <sstream>


using namespace cv;
using namespace pdi;
using namespace std;
using namespace fs;
using namespace utils;

Mat esta_Rotada(Mat img){
//	imshow("Imagen Original",img);
	Mat roi = img(Rect(197,30,3,3));
//	MostrarHistogramas(roi);
	Mat img_hsv;
	cvtColor(img,img_hsv,CV_BGR2HSV);
//	vector<Mat> rgb; 	
//	split(img, rgb);
	Mat gris = img.clone();
	cvtColor(gris,gris,CV_BGR2GRAY);
	Mat mascara = cv::Mat::zeros(img.size(),gris.type());
	Mat mascara2 = cv::Mat::zeros(img.size(),gris.type());
	
	Point center(img.cols/2, img.rows/2);
	int radio = 165;
	// circle center
	circle(mascara,center,radio, 255, 15, 8, 0 );
	imshow("mascara",mascara);
	for(int i=0;i<img.rows;i++){ 
		for(int j=0;j<img.cols;j++) { 
			if(img_hsv.at<Vec3b>(i,j)[2] > 240){
				if(img.at<Vec3b>(i,j)[2] > 240){
					if(mascara.at<uchar>(i,j) == 255){
						mascara2.at<uchar>(i,j) = 255;
					}
				}
			}
		}
	}
	Mat EE = getStructuringElement(MORPH_RECT,Size(2,2));
	erode(mascara2,mascara2,EE);
	Mat EE2 = getStructuringElement(MORPH_RECT,Size(5,5));
	dilate(mascara2,mascara2,EE2);
	
	int ii=0;
	int jj=0;
	for(int i=0;i<mascara2.rows;i++) { 
		for(int j=0;j<mascara2.cols;j++) { 
			if (mascara2.at<uchar>(i,j)==255 && ii==0 && jj==0){
				ii=i;
				jj=j;
			}
			
		}
	}
	double angulo;
	int angulo_final;
	if (jj<(int)mascara.cols/2 -10){
		double a = abs(mascara.cols/2-jj);
		double b = abs(mascara.rows/2-ii);
		double division = a/b;
		angulo = atan(division)*180/M_PI;
		if(angulo < 3){
			angulo = 0;
		}
		angulo_final = round(angulo/5);
		angulo_final = angulo_final*5;
		img = rotate(img,-angulo_final);
	}
	else{
		double a = abs(jj-mascara.cols/2);
		double b = abs(mascara.rows/2-ii);
		double division = a/b;
		angulo = atan(division)*180/M_PI;
		if(angulo < 3){
			angulo = 0;
		}
		angulo_final = round(angulo/5);
		angulo_final = angulo_final*5;
		img = rotate(img,angulo_final);
	}
//	cout<<angulo_final<<endl;
//	imshow("Mascara",mascara);
//	imshow("Mascara2",mascara2);
//	imshow("Imagen Rotada",img);
//	waitKey(0);
	return img;
}
void Brujulas(Mat img){
	Mat aux = img.clone();
	Mat gris = img.clone();
	cvtColor(gris,gris,CV_BGR2GRAY);

	
	Mat roi = img(Rect(197,30,5,5));
//	MostrarHistogramas(roi);
	
	vector<Mat> rgb2; 	
	split(roi, rgb2);
	float media1 = obtenerMedia(rgb2[0]);
	float media2 = obtenerMedia(rgb2[1]);
	float media3 = obtenerMedia(rgb2[2]);
	float desvio1 = obtenerDesvio(rgb2[0]);
	float desvio2 = obtenerDesvio(rgb2[1]);
	float desvio3 = obtenerDesvio(rgb2[2]);
	
	for (int i=0;i<img.rows;i++){
		for (int j=0;j<img.cols;j++){
			if (j>185 && j<216){ //Pinto el tirangulito
				if(i>20 && i<50){
					img.at<Vec3b>(i,j) = {0,0,0};
				}
			}
			if (j>185 && j<216){	//Pinto la N
				if(i>80 && i<130){
					img.at<Vec3b>(i,j) = {0,0,0};
				}
			}
		}
	}
	vector<Mat> bgr1; 	
	split(img, bgr1);
	
	Mat mascara = cv::Mat::zeros(aux.size(),gris.type());
	for(int i=0;i<img.rows;i++){ 
		for(int j=0;j<img.cols;j++) { 
			if ((bgr1[0].at<uchar>(i,j) > media1-desvio1) && (bgr1[0].at<uchar>(i,j) < media1+desvio1)){
				if ((bgr1[1].at<uchar>(i,j) > media2-desvio2) && (bgr1[1].at<uchar>(i,j) < media2+desvio2)){
					if ((bgr1[2].at<uchar>(i,j) > media3-desvio3) && (bgr1[2].at<uchar>(i,j) < media3+desvio3)){
						mascara.at<uchar>(i,j) = 255;
					}
				}
			}
		}
	}
	
	Mat kernel = Filtro_Promediador(3);
	mascara = convolve(mascara,kernel);
	for (int i=0;i<mascara.rows;i++){
		for (int j=0;j<mascara.cols;j++){
			if ((int)mascara.at<uchar>(i,j)>10){
				mascara.at<uchar>(i,j) = 255;
			}
			else{
				mascara.at<uchar>(i,j) = 0;
			}
		}
	}
	Mat kernel2 = Filtro_Promediador(3);
	mascara = convolve(mascara,kernel2);
	for (int i=0;i<mascara.rows;i++){
		for (int j=0;j<mascara.cols;j++){
			if ((int)mascara.at<uchar>(i,j)>10){
				mascara.at<uchar>(i,j) = 255;
			}
			else{
				mascara.at<uchar>(i,j) = 0;
			}
		}
	}
	Mat EE = getStructuringElement(MORPH_RECT,Size(20,20));
	dilate(mascara,mascara,EE);
//	Mat EE2 = getStructuringElement(MORPH_RECT,Size(10,10));
//	erode(mascara,mascara,EE2);
//	Mat EE2 = getStructuringElement(MORPH_RECT,Size(5,5));
//	erode(mascara,mascara,EE2);
	
	imshow("Mascara",mascara);
	int ii=0;
	int jj=0;
	for(int i=0;i<mascara.rows;i++) { 
		for(int j=0;j<mascara.cols;j++) { 
			if (mascara.at<uchar>(i,j)==255 && ii==0 && jj==0){
				ii=i;
				jj=j;
			}
			
		}
	}
	mascara.convertTo(mascara,CV_32F,1./255);
	Mat espectro = spectrum(mascara);
//	imshow("Espectro",espectro);
	
	Mat umbral(espectro.size(),espectro.type());
	for(int i=0;i<espectro.rows;i++) { 
		for(int j=0;j<espectro.cols;j++) { 
			if (espectro.at<float>(i,j) > 130/255.0){
				umbral.at<float>(i,j) = 1;
			}
			else umbral.at<float>(i,j) = 0;
		}
	}
	//	namedWindow("Umbral 1",CV_WINDOW_KEEPRATIO);
	//	imshow("Umbral 1",umbral);
	
	Mat kernel3 = Filtro_Promediador(5);
	umbral = convolve(umbral,kernel3);
	for(int i=0;i<umbral.rows;i++) { 
		for(int j=0;j<umbral.cols;j++) { 
			if (umbral.at<float>(i,j)>80/255.0){
				umbral.at<float>(i,j) = 1;
			}
			else{
				umbral.at<float>(i,j) = 0;
			}
		}
	}
	namedWindow("Umbral 2",CV_WINDOW_KEEPRATIO);
	imshow("Umbral 2",umbral);
	
	double angulo;
	if (jj<(int)mascara.cols/2 -10){
		double a = abs(mascara.cols/2-jj);
		double b = abs(mascara.rows/2-ii);
		double division = a/b;
		angulo = atan(division)*180/M_PI;
		angulo = angulo + 180;
	}
	else{
		double a = abs(jj-mascara.cols/2);
		double b = abs(mascara.rows/2-ii);
		double division = a/b;
		angulo = atan(division)*180/M_PI;
		angulo = 180 - angulo;
	}
	
	cout<<angulo<<endl;
	namedWindow("Imagen original",CV_WINDOW_KEEPRATIO);
	imshow("Imagen original",aux);
	waitKey(0);
}
int main(int argc, char** argv) {
	for(int i=1;i<5;i++) { 
		string aux="Brujulas/";
		string nombre;
		stringstream c;
		c<<i;
		nombre=c.str();
		aux=aux+nombre+".png";
		Mat img=imread(aux);
		img = esta_Rotada(img);
		Brujulas(img);
	}
	return 0;
} 
