#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <opencv2/core/core.hpp>



using namespace cv;
using namespace pdi;
using namespace std;

///La funcion de abajo sirve para averiguar la posicion y componente rgb de un pixel en la imagen que se define en este 
///ambito

/*
void onMouse( int event, int x, int y, int, void* );

Mat imagen = imread("rostro10.png");

void onMouse( int event, int x, int y, int, void* )
{
	if( event != CV_EVENT_LBUTTONDOWN )
		return;
	
	Point pt = Point(x,y);
	cout<<"x="<<pt.x<<"\t y="<<pt.y;
	Vec3b valores=imagen.at<cv::Vec3b>(y,x);
	cout<<"\t B="<<(int)valores.val[0]<<" G="<<(int)valores.val[1]<<" R="<<(int)valores.val[2]<<endl;
	
}

void Averiguar(){
	
	namedWindow("Averiguar Pixeles");
	setMouseCallback( "Averiguar Pixeles", onMouse, 0 );
	imshow("Averiguar Pixeles",imagen);
	
}*/


//Recordar que el [0] corresponde al azul, [1] al verde y [2] al rojo
vector<Mat> planosRGB(Mat img){
	vector<Mat>bgr;
	split(img,bgr);
	return bgr;
}

vector<Mat> planosHSV(Mat img){
	cvtColor(img,img,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img,hsv);
	return hsv;
}

Mat planosHSVtoimagenRGB(vector<Mat> hsv){
	Mat img_hsv,img_rgb;
	merge(hsv,img_hsv);
	cvtColor(img_hsv, img_rgb, CV_HSV2BGR);
	return img_rgb;
}

Mat equalizarRGB(Mat img){
	vector<Mat>bgr;
	split(img,bgr);
	equalizeHist(bgr[0],bgr[0]);
	equalizeHist(bgr[1],bgr[1]);
	equalizeHist(bgr[2],bgr[2]);
	merge(bgr,img);
	return img;
}

Mat equalizarHSV(Mat img){
	cvtColor(img, img,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img,hsv);
	//Solo se ecualiza el canal V
	equalizeHist(hsv[2],hsv[2]);
	merge(hsv,img);
	//Vuelvo a RGB
	cvtColor(img,img,CV_HSV2BGR);
	return img;
	
}

///Cuando esten integradas todas las funciones, reemplazar por la rutina aplicar_filtro_suma_1
///REVEER TODA LA FUNCION realceMedianteAcentuadoRGB

Mat filtro_pasa_alto_suma1(int tam){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam;
	return kernel;
}

Mat realceMedianteAcentuadoRGB(Mat img){
	Mat kernel = filtro_pasa_alto_suma1(3);
	vector<Mat> bgr;
	split(img,bgr);
	bgr[0] = convolve(bgr[0],kernel);
	bgr[1] = convolve(bgr[1],kernel);
	bgr[2] = convolve(bgr[2],kernel);
	merge(bgr,img);
	return img;	
}

Mat realceMedianteAcentuadoHSV(Mat img){
	Mat kernel = filtro_pasa_alto_suma1(3);
	cvtColor(img,img,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img,hsv);
	//El kernel se aplica solo sobre el plano de brillo o valor
	hsv[2]=convolve(hsv[2],kernel);
	merge(hsv,img);
	cvtColor(img,img,CV_HSV2BGR);
	return img;
}
