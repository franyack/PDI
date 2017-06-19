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

int AnillosOro(Mat img){
//	Mat img= imread(imagen);
//	namedWindow("original",CV_WINDOW_KEEPRATIO);
//	imshow("original",img);
	
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
	inRange(img,Scalar(152,219,234),Scalar(163,227,255),mascara);
	
	
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
//	imshow("mascara", mascara);
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchyOro;
	findContours(mascara, contornos, hierarchyOro, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//	cout<<"Numero de anillos de Oro: "<<hierarchyOro.size()<<endl;
	return hierarchyOro.size();
}

int AnillosPlata(Mat img){
//	Mat img= imread(imagen);
//	namedWindow("original",CV_WINDOW_KEEPRATIO);
//	imshow("original",img);
	
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
	inRange(img,Scalar(220,220,220),Scalar(230,230,230),mascara);
	
	
	erode(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	
	
//	imshow("mascara", mascara);
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchyOro;
	findContours(mascara, contornos, hierarchyOro, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//	cout<<"Numero de anillos de Plata: "<<hierarchyOro.size()<<endl;
	return hierarchyOro.size();
}


void Parcialanillos(string imagen){
	Mat img = imread(imagen);
	namedWindow("original",CV_WINDOW_KEEPRATIO);
	imshow("original",img);
	int oro = AnillosOro(img);
	int plata = AnillosPlata(img);
	cout<<"Total anillos de oro: "<<oro<<endl;
	cout<<"Total anillos de plata: "<<plata<<endl;
	cout<<"Total de anillos: "<<oro + plata<<endl;
	
	
	///MEñique 1 y 2 - media > 30
	Mat roi=img(Rect(0,0,img.cols/4,img.rows));
//	imshow("ss",roi);
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
	inRange(roi,Scalar(0,0,0),Scalar(250,250,250),mascara);
//	imshow("ss",mascara);
	float media = obtenerMedia(mascara);
//	cout<<"media "<<media<<endl;
	if(media > 30){
		oro = AnillosOro(roi);
		plata = AnillosPlata(roi);
		cout<<"El dedo meñique tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	}
	
	///Anular 1 y 2 - media > 100 -------- Indice 3 media < 100
	Mat roi2=img(Rect(img.cols/4,0,img.cols/6,img.rows/2.1));
//	imshow("ss",roi2);
	inRange(roi2,Scalar(0,0,0),Scalar(250,250,250),mascara);
	media = obtenerMedia(mascara);
//	cout<<"media "<<media<<endl;
//	imshow("ss",mascara);
	oro = AnillosOro(roi2);
	plata = AnillosPlata(roi2);
	if(media > 100){
		cout<<"El dedo anular tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	}else{
		cout<<"El dedo indice tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	}
	
	///Mayor - queda en el mismo lugar para todas las imagenes
	Mat roi3=img(Rect(img.cols/2.4,0,50,img.rows/2));
//	imshow("ss",roi3);
	inRange(roi3,Scalar(0,0,0),Scalar(250,250,250),mascara);
	erode(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	media = obtenerMedia(mascara);
//	cout<<"media "<<media<<endl;
//	imshow("mascara",mascara);
	oro = AnillosOro(roi3);
	plata = AnillosPlata(roi3);
	cout<<"El dedo mayor tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	
	
	///Indice 1 y 2 - media < 110 -------- Anular 3 media > 110
	Mat roi4=img(Rect(img.cols/1.7,0,img.cols/7.5,img.rows/2));
//	imshow("ss",roi4);
	inRange(roi4,Scalar(0,0,0),Scalar(250,250,250),mascara);
	erode(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	media = obtenerMedia(mascara);
//	cout<<"media "<<media<<endl;
//	imshow("mascara",mascara);
	oro = AnillosOro(roi4);
	plata = AnillosPlata(roi4);
	if(media < 110){
		cout<<"El dedo indice tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	}else{
		cout<<"El dedo anular tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	}
	
	
	///Meñique imagen 3
	Mat roi5=img(Rect(img.cols/1.34,0,img.cols/7.5,img.rows));
//	imshow("ss",roi5);
	inRange(roi5,Scalar(0,0,0),Scalar(250,250,250),mascara);
	erode(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	media = obtenerMedia(mascara);
//	cout<<"media "<<media<<endl;
//	imshow("mascara",mascara);
	oro = AnillosOro(roi5);
	plata = AnillosPlata(roi5);
	if(media > 60){
		cout<<"El dedo meñique tiene: "<<oro<<" anillos de oro y "<<plata<<" anillos de plata"<<endl;
	}
}

int main(int argc, char** argv) {

	string imagen = "1.png";
	Parcialanillos(imagen);

	waitKey(0);
	return 0;
} 
