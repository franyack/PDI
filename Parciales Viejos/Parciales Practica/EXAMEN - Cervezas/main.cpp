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
	
	Mat img = imread("34.jpg");
	//Primero saco las transformada Hough para obtener las lineas del vaso.
	
	Mat Gradiente2;
	
	Canny(img,Gradiente2,50,200,3); //Detecto los bordes de la imagen
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	imshow("Original",img);
	namedWindow("Bordes",CV_WINDOW_KEEPRATIO);
	imshow("Bordes",Gradiente2);
	
	
	//HoughLinesP me da como salida los extremos de las lineas detectadas. (x0,y0),(x1,y1).
	vector <Vec4i> lines2;
	HoughLinesP(Gradiente2,lines2,1,CV_PI/180, 10, 10, 10  ); 
	Mat transformadaP;
	cvtColor(Gradiente2, transformadaP, CV_GRAY2BGR);

	//Los parametros son los mismos, pero los ultimos dos son:
	//* El numero minimo de puntos que se puede formar una linea, las lineas con menos de estos puntos no se tienen en cuenta
	//* Separacion maxima entre dos puntos a considerar en la misma recta
	for(size_t i=0; i<lines2.size(); i++) { 
		Vec4i  l=lines2[i];
		line(transformadaP ,Point(l[0],l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA); 
	}
	namedWindow("Transformada HoughP",CV_WINDOW_KEEPRATIO);
	imshow("Transformada HoughP",transformadaP);
	//Una vez obtenidas las lineas de los contornos, de los mismos obtengo el menor xy y mayor xy de cada vaso para hacer un roi de solo el vaso
	int minx=9999;
	int miny=9999;
	int maxx=-1;
	int maxy=-1;
	for(int i=0;i<lines2.size();i++) { 
		Vec4i  v=lines2[i];
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
	
	
	Mat auxiliar=img.clone();
	Mat ROI=auxiliar(Rect(minx,miny,maxx-minx,maxy-miny));
	namedWindow("ROI",CV_WINDOW_KEEPRATIO);
	imshow("ROI",ROI);
	
	//Armo un segundo roi del roi que ya tenia con un cuadrado central y le saco la media (si es rubia la media da 150 mas o menos
	//si es negra la media da entre 50 y 90 depende la iluminacion, entonces con preguntar de la media nomas ya se cual es).
	Mat roi2=ROI(Rect(ROI.cols/2-10,ROI.rows/2-10,20,20));
	//	imshow("2do ROI",roi2);
	float media=obtenerMedia(roi2);
	if (media>100){
		cout<<"Es una Cerveza RUBIA"<<endl;
	}
	else{
		cout<<"Es una Cerveza NEGRA"<<endl;
	}
	
	//Con el canal B es el mejor para determinar la espuma, recorro por la linea central y si es mayor a 110 voy sumando espuma
	//como todos los vasos tienen una base esta la cuenta como espuma tmb, entonces pregunto si el valos de B es blanco y sus
	//anteriores no lo eran, entonces corto. Y luego para sacar el promedio de espuma lo hago hasta el valor donde encontro que 
	//empieza la base
	vector <Mat> bgr; 	
	bgr = planosRGB(ROI);
	int espuma=0;
	for(int i=0;i<ROI.rows;i++) { 
		if ((int)bgr[0].at<uchar>(i,ROI.cols/2)>130){
			espuma++;
			if (i>6 && (int)bgr[0].at<uchar>(i-5,ROI.cols/2)<60){
				break;
			}
		}
//		cout<<(int)bgr[0].at<uchar>(i,ROI.cols/2)<<endl;
	}

	float porcentaje=espuma*100/ROI.rows;
	cout<<"La cerveza tiene un porcentaje de espuma del: "<<porcentaje<<"%"<<endl;
	namedWindow("Finaly",CV_WINDOW_KEEPRATIO);
	imshow("Finaly",ROI);
	
	
	waitKey(0);
	return 0;
} 
