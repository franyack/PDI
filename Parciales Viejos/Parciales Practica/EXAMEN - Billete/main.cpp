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
	//Primero verifico si la imagen esta derecha o rotada 180º. para esto con perfiles de intensidad
	//elijo cuatro lineas fijas y saco el promedio de gris de cada una y un promedio general de los
	//cuatro promedios elegidos, por eso si la img tiene un promedio menor a 200 hay que rotar 180º
	Mat img = imread("B50C1_01a.jpg");
	Mat aux=img.clone();
	cvtColor(aux,aux,CV_BGR2GRAY);
	float promedio=0.0;
	float promedio1=0.0;
	float promedio2=0.0;
	float promedio3=0.0;
	for(int i=0;i<aux.rows;i++) { 
		promedio+=(int)aux.at<uchar>(i,110);
		promedio1+=(int)aux.at<uchar>(i,115);
		promedio2+=(int)aux.at<uchar>(i,120);
		promedio3+=(int)aux.at<uchar>(i,125);
	}
	promedio/=aux.rows;
	promedio1/=aux.rows;
	promedio2/=aux.rows;
	promedio3/=aux.rows;
	float PromedioGeneral=(promedio+promedio1+promedio2+promedio3)/4;
	cout<<"Promedio: "<<PromedioGeneral<<endl;
	if (PromedioGeneral<200){
		aux=rotate(aux,180);
	}
	
	//Para saber que billete es saco los triangulitos, y como todas las imagenes van a ser siempre del mismo tamaño
	//y van a estar en la misma direccion (ya fueron rotadas) los triangulos van a estar en el mismo lugar.
	//Entonces saco un roi que comprenda todos los triangulos de cada billete
	
	Mat roi=aux(Rect(132,13,50,75));
	imshow("ss",roi);
	
	//Umbralizo
	Mat Mascara=cv::Mat::zeros(roi.size(),roi.type());
	
	for(int i=0;i<roi.rows;i++) { 
		for(int j=0;j<roi.cols;j++) { 
			if (roi.at<uchar>(i,j)<120){
				Mascara.at<uchar>(i,j)=255;
			}
		}
	}
	
	//Con el dilate directamente tapo los huecos
	Mat EE=getStructuringElement(MORPH_RECT,Size(3,3));
	dilate(Mascara,Mascara,EE);
	
	/*	Mascara = aplicar_filtro_promediador(Mascara);*/
	
	imshow("asd",Mascara);
	
	//Ahora uso la funcion findContours para contar los triangulitos.
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchy;
	Mat mask=Mascara.clone();
	findContours(mask, contornos, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	int triangulos=hierarchy.size();
	switch (triangulos){
	case 1:
		cout<<"El billete presentado es de $100"<<endl;
		break;
	case 2:
		cout<<"El billete presentado es de $50"<<endl;
		break;
	case 3:
		cout<<"El billete presentado es de $20"<<endl;
		break;
	case 4:
		cout<<"El billete presentado es de $10"<<endl;
		break;
	case 5:
		cout<<"El billete presentado es de $5"<<endl;
		break;
	case 6:
		cout<<"El billete presentado es de $2"<<endl;
		break;
	default: 
		cout<<"ERROR"<<endl;
		break;
	}
	
	waitKey(0);
	return 0;
} 
