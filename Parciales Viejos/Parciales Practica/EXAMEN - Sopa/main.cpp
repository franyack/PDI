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
	
	Mat img = imread("1.jpg");
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	imshow("Original",img);
	
	//Aplico la transformada de hough circular y obtengo el centro del plato
	Mat gris=img.clone();
	cvtColor(gris,gris,CV_BGR2GRAY);
	GaussianBlur( gris, gris, Size(9, 9), 2, 2 );
	
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles
	HoughCircles( gris, circles, CV_HOUGH_GRADIENT, 2, gris.rows/8, 200, 100, 0, 0 );
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( gris, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( gris, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}
	
	/// Show your results
//		namedWindow( "Hough Circle Transform Demo", CV_WINDOW_KEEPRATIO );
//		imshow( "Hough Circle Transform Demo", gris );
	//DETECTAR TIPO DE SOPA
	Mat roi=img(Rect(circles[0](0)-20,circles[0](1)-20,40,40));
	imshow("ROI",roi);
	
	Mat roiaux=roi.clone();
	vector <Mat> hsv;
 	hsv = planosHSV(roiaux);
	
	float media=obtenerMedia(hsv[0]);
	string sopa;
	if (media>17){
		sopa="la casa.";
	}
	else{
		sopa="zapallo.";
	}
	
	int escena=0;
	//DETECTAR TOTAL DE MOSCAS EN LA ESCENA
	Mat aux=img.clone();
	cvtColor(aux,aux,CV_BGR2GRAY);
//	namedWindow("Gris",CV_WINDOW_KEEPRATIO);
//	imshow("Gris",aux);
	
	Mat mascara=cv::Mat::zeros(aux.size(),aux.type());
	
	for(int i=0;i<aux.rows;i++) { 
		for(int j=0;j<aux.cols;j++) { 
			if (aux.at<uchar>(i,j)<10){
				mascara.at<uchar>(i,j)=255;
			}
		}
	}
	
	
	mascara = aplicar_filtro_promediador(mascara);

	
	for (int i=0;i<mascara.rows;i++){
		for (int j=0;j<mascara.cols;j++){
			if ((int)mascara.at<uchar>(i,j)>220){mascara.at<uchar>(i,j)=255;}
			else{mascara.at<uchar>(i,j)=0;}
		}
	}
	
	
	Mat EE=getStructuringElement(MORPH_RECT,Size(9,9));
	dilate(mascara,mascara,EE);
	
	Mat mascaraaux=mascara.clone();
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchy;
	findContours(mascara, contornos, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	escena=hierarchy.size();
	cout<<"El numero total de moscas en la escena es de: "<<escena<<endl;
//		namedWindow("Mascara1",CV_WINDOW_KEEPRATIO);
//		imshow("Mascara1",mascaraaux);
	
	//DETECTAR MOSCAS DENTRO DEL PLATO
	Mat aux2=img.clone();
	cvtColor(aux2,aux2,CV_BGR2GRAY);
	
	//Eso me hace una mascara binaria donde lo negro es lo de afuera del plato, y lo blanco todo el plato
	Mat mascara2=cv::Mat::zeros(aux2.size(),aux2.type());
	circle(mascara2, Point(circles[0](0),circles[0](1)), circles[0](2), Scalar(255), -1, 8, 0);
//	mascara2 = aplicar_filtro_promediador(mascara2);
//	imshow("sa1",mascara2);
	
	//En esta parte vuelvo a pintar las moscas, las que estan afuera del plato ya estaban negras, las 
	//de adentro del plato se pintan de negro
	
	for(int i=0;i<aux2.rows;i++) { 
		for(int j=0;j<aux2.cols;j++) { 
			if (aux2.at<uchar>(i,j)<10){
				mascara2.at<uchar>(i,j)=0;
			}
		}
	}
	mascara2 = aplicar_filtro_promediador(mascara2);
	
	//lo hago para invertir y sacar el dilate
	for(int i=0;i<mascara2.rows;i++) { 
		for(int j=0;j<mascara2.cols;j++) { 
			if ((int)mascara2.at<uchar>(i,j)==0){mascara2.at<uchar>(i,j)=255;}
			else{mascara2.at<uchar>(i,j)=0;} 
		}
	}
	
	////	
	Mat EE2=getStructuringElement(MORPH_RECT,Size(9,9));
	dilate(mascara2,mascara2,EE2);

	
	//	
	//Ahora cuento cuantas moscas hay en el plato
	vector<vector<Point> > contornos2;
	vector<Vec4i> hierarchy2;
	findContours(mascara2, contornos2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	int plato=0;
	plato=hierarchy2.size()-2; //le resto dos porque me cuenta dos contornos de mas que son el plato y el fondo.
	cout<<"El numero total de moscas en el plato es de: "<<plato<<endl;
//		namedWindow("Mascara",CV_WINDOW_KEEPRATIO);
//		imshow("Mascara",mascara2);
//		cout<<circles[0](2)<<endl;
	
	//VER CUANTAS HAY EN LA SOPA
	Mat aux3=img.clone();
	cvtColor(aux3,aux3,CV_BGR2GRAY);

	Mat mascara3=cv::Mat::zeros(aux3.size(),aux3.type());
	circle(mascara3, Point(circles[0](0),circles[0](1)), circles[0](2)-150, Scalar(255), -1, 8, 0);

	for(int i=0;i<aux3.rows;i++) { 
		for(int j=0;j<aux3.cols;j++) { 
			if (aux3.at<uchar>(i,j)<10){
				mascara3.at<uchar>(i,j)=0;
			}
		}
	}
	
	
	mascara3=aplicar_filtro_promediador(mascara3);

	//lo hago para invertir y sacar el dilate
	for(int i=0;i<mascara3.rows;i++) { 
		for(int j=0;j<mascara3.cols;j++) { 
			if ((int)mascara3.at<uchar>(i,j)==0){mascara3.at<uchar>(i,j)=255;}
			else{mascara3.at<uchar>(i,j)=0;} 
		}
	}
	////	
	Mat EE3=getStructuringElement(MORPH_RECT,Size(9,9));
	dilate(mascara3,mascara3,EE3);


	vector<vector<Point> > contornos3;
	vector<Vec4i> hierarchy3;
	findContours(mascara3, contornos3, hierarchy3, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	int sope=0;
	sope=hierarchy3.size()-2; //le resto dos porque me cuenta dos contornos de mas que son el plato y el fondo.
	cout<<"El numero total de moscas en la sopa es de: "<<sope<<endl;

	if ((sopa=="zapallo." && sope<4) || (sopa=="la casa." && sope<5)){
		cout<<"EL PLATO ESTA BIEN SERVIDO!"<<endl<<endl;
	}
	else{
		cout<<"EL PLATO ESTA MAL SERVIDO!"<<endl<<endl;
	}
	
	
	waitKey(0);
	return 0;
} 
