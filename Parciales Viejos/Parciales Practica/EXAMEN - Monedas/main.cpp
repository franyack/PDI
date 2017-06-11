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
	
	Mat img = imread("01_Monedas.jpg");
	
	Mat aux=img.clone();
	cvtColor(aux,aux,CV_BGR2GRAY);
	Mat mascara=cv::Mat::zeros(aux.size(),aux.type());
	for(int i=0;i<img.rows;i++) { 
		for(int j=0;j<img.cols;j++) { 
			if (aux.at<uchar>(i,j)<230){
				mascara.at<uchar>(i,j)=255;
			}
		}
	}
	mascara = aplicar_filtro_promediador(mascara,9);
	
	for (int i=0;i<mascara.rows;i++){
		for (int j=0;j<mascara.cols;j++){
			if ((int)mascara.at<uchar>(i,j)>150){mascara.at<uchar>(i,j)=255;}
			else{mascara.at<uchar>(i,j)=0;}
		}
	}
	
	Mat auxmascara=mascara.clone();
	vector<vector<Point> > contornos;
	vector<Vec4i> hierarchy;
	findContours(auxmascara, contornos, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	cout<<"El numero total de monedas es de: "<<hierarchy.size()<<endl;
	float total=0.0;
	
	for(int i=0;i<contornos.size();i++) { 
		float area=contourArea(contornos[i],false);
		if (area>2800 && area<3000){
			cout<<"Moneda de 1 centavo"<<endl;
			total+=0.01;
		}
		else{
			if (area>3800 && area<4000){
				cout<<"Moneda de 2 centavos"<<endl;
				total+=0.02;
			}
			else{
				if (area>4200 && area<4400){
					cout<<"Moneda de 10 centavos"<<endl;
					total+=0.10;
				}
				else{
					if(area>5000 && area<5200){
						cout<<"Moneda de 5 centavos"<<endl;
						total+=0.05;
					}
					else{
						if(area>5400 && area<5600){
							cout<<"Moneda de 20 centavos"<<endl;
							total+=0.20;
						}
						else{
							if(area>5900 && area<6100){
								cout<<"Moneda de 1 euro"<<endl;
								total+=1.0;
							}
							else{
								if(area>6500 && area<6750){
									cout<<"Moneda de 50 centavos"<<endl;
									total+=0.50;
								}
								else{
									if(area>7300 && area<7600){
										cout<<"Moneda de 2 euros"<<endl;
										total+=2.0;
									}
									else{
										cout<<"ERROR PAJERO"<<endl;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	cout<<"El total de dinero que hay es de: "<<total<<" euros."<<endl<<endl;
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	imshow("Original",img);
	
	namedWindow("Mascara",CV_WINDOW_KEEPRATIO);
	imshow("Mascara",mascara);

	waitKey(0);
	return 0;
} 
