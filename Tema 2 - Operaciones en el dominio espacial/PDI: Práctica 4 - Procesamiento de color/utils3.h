#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace pdi;
using namespace std;

Mat histograma(Mat img){
	Mat canvas(img.rows,img.cols,CV_32F);
	Mat histo = histogram(img,255);
	normalize(histo,histo,0,1,CV_MINMAX);
	draw_graph(canvas,histo);
	return canvas;
}

Mat ecualizarImagen(Mat img){
	Mat ecualizado;
	equalizeHist(img,ecualizado);
	return ecualizado;
}

Mat histogramaEcualizado(Mat img){
	Mat ecualizado;
	equalizeHist(img,ecualizado);
	Mat canvas(ecualizado.rows,ecualizado.cols,CV_32F);
	Mat histo = histogram(ecualizado,255);
	normalize(histo,histo,0,1,CV_MINMAX);
	draw_graph(canvas,histo);
	return canvas;
}



Mat aplicar_filtro_promediador(Mat img, int tam=3){
	Mat kernel(tam,tam,CV_32F);
	float aux=1/pow(tam,2);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=aux;
		}
	}
	Mat f = convolve(img,kernel);
	return f;
}

Mat aplicar_filtro_gaussiano(Mat img, int tam=3, float sigma = 1){
	Mat f;
	GaussianBlur(img,f,Size(tam,tam),sigma);
	return f;
}

Mat aplicar_filtro_suma1(Mat img, int tam=3){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam;
	
	Mat result = convolve(img,kernel);
	return result;
}

Mat aplicar_filtro_suma0(Mat img, int tam = 3){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam-1;
	
	Mat result = convolve(img,kernel);
	return result;
}

Mat aplicar_filtro_altaPotencia(Mat img, float A = 1, int tam=3){
	// Si A = 1 -> Mascara Difusa, sino, Alta Potencia
	Mat pb = aplicar_filtro_promediador(img,tam);
	
	Mat result = (A * img - pb + 255)/2;
	
	return result;
	
}

