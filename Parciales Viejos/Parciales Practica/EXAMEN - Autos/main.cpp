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

Point2f dibujarlinea(Mat img){
	
	Mat aux = img.clone();
	
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
	inRange(img,Scalar(155,102,200),Scalar(165,115,215),mascara);
	
	imshow("mascara", mascara);
	
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchy;
	findContours(mascara, contornos, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//			cout<<"Numero de contornos: "<<hierarchy.size()<<endl;
//	
	vector<Moments> mu(contornos.size() );
	for( size_t i = 0; i < contornos.size(); i++ )
	{ mu[i] = moments( contornos[i], false ); }
	
	//  A partir de los momentos se pueden hallar los centros de masa
	vector<Point2f> mass( contornos.size() );
	for( size_t i = 0; i < contornos.size(); i++ ){
		mass[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
	}
	Point2f masa_libre = mass[0];
	return masa_libre;
}


void ParcialEstacionamiento(string imagen){
	Mat img = imread(imagen);
	imshow("kajsd",img);
//	waitKey(0);
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
//	namedWindow("original",CV_WINDOW_KEEPRATIO);
//	imshow("original",img);
//	waitKey(0);
	inRange(img,Scalar(60,130,180),Scalar(90,170,210),mascara);
	Mat aux2;
	img.copyTo(aux2,mascara);
	
	erode(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	dilate(mascara,mascara,getStructuringElement(MORPH_RECT,Size(6,6)));
	bool estalibre = 0;
//	imshow("erode",mascara);
//	waitKey(0);
	float suma = 0;
	for(int i=0;i<mascara.cols;i++) { 
		suma+=mascara.at<uchar>(mascara.rows/4,i);
	}
	suma/=mascara.rows;
//	cout<<"suma "<<suma<<endl;
	if(suma < 70){
		cout<<"El estacionamiento esta ocupado"<<endl;
	}else{
		cout<<"El estacionamiento esta libre"<<endl;
		estalibre = 1;
		
	}
	
	Mat aux = mascara.clone();
	
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchy;
	findContours(aux, contornos, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	//		cout<<"Numero de contornos: "<<hierarchy.size()<<endl;
	
	vector<Moments> mu(contornos.size() );
	for( size_t i = 0; i < contornos.size(); i++ )
	{ mu[i] = moments( contornos[i], false ); }
	
	//  A partir de los momentos se pueden hallar los centros de masa
	vector<Point2f> mass( contornos.size() );
	for( size_t i = 0; i < contornos.size(); i++ ){
		mass[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
	}
	Point2f masa_libre = mass[0];
	
	///Si estan a 45, la media es 118, si estan a 90, 131
	//calculos para 45 grados
	if(estalibre && suma<126){
		Point2f masa = dibujarlinea(img);
		line(img ,masa_libre, masa, Scalar(0,0,255),5,8,0);
		if(masa_libre.x < (int)img.cols/4){
			cout<<"Esta en el primer lugar";
		}else{
			if(masa_libre.x>(int)img.cols/4 && masa_libre.x<(int)img.cols/2){
				cout<<"Esta en el segundo lugar";
			}else{
				if(masa_libre.x>(int)img.cols/2 && masa_libre.x<(int)img.cols*3/4){
					cout<<"Esta en el tercer lugar";
				}else{
					cout<<"Esta en el cuarto lugar";
				}
			}
		}
	}
	if(estalibre && suma > 126){
		Point2f masa = dibujarlinea(img);
		line(img ,masa_libre, masa, Scalar(0,0,255), 3, CV_AA); 
		if(masa_libre.x<(int)img.cols/4){
			cout<<"Esta en el primer lugar";
		}else{
			if(masa_libre.x>(int)img.cols/4 && masa_libre.x<(int)img.cols/2){
				cout<<"Esta en el segundo lugar";
			}else{
				if(masa_libre.x>(int)img.cols/2 && masa_libre.x<(int)img.cols*3/4){
					cout<<"Esta en el tercer lugar";
				}else{
					if(masa_libre.x>(int)img.cols*3/4 && masa_libre.x<(int)img.cols){
						cout<<"Esta en el cuarto lugar";
					}
				}
			}
		}
	}
	imshow("imagen final",img);

}

int main(int argc, char** argv) {
	
	ParcialEstacionamiento("parking5.png");
	

	waitKey(0);
	return 0;
} 
