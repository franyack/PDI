#ifndef FUNCIONES_FS_H
#define FUNCIONES_FS_H
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include "pdi_functions.h"
#include "funciones_FS.h"
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
using namespace std;
using namespace cv;
using namespace pdi;

namespace fs{
	void onMouse( int event, int x, int y, int, void* );
	//
	Mat imagen = imread("Billetes/1.jpg");
	void onMouse( int event, int x, int y, int, void* )
	{
		if( event != CV_EVENT_LBUTTONDOWN )
			return;
		
		Point pt = Point(x,y);
		cout<<"x="<<pt.x<<"\t y="<<pt.y;
		Vec3b valores=imagen.at<cv::Vec3b>(y,x);
		cout<<"\t B="<<(int)valores.val[0]<<" G="<<(int)valores.val[1]<<" R="<<(int)valores.val[2]<<endl;
		
	}
	//Con esta funcion averiguo el x y de un pixel que hago click y sus valores en R, G y B.
	//Debo cambiar aca arriba la imagen que quiero saber los valores.
	void Averiguar(){
		namedWindow("Averiguar Pixeles");
		setMouseCallback( "Averiguar Pixeles", onMouse, 0 );
		imshow("Averiguar Pixeles",imagen);
		
	}
	
	
	//GUIA 2
	
	//A MEDIDA QUE VOY AUMENTANDO LA VARIABLE INDEPENDIENTE ES DECIR EL C, EL 
	//GRAFICO DE MAPEO SE ENCUENTRA DESPLAZADO, DONDE TODO LO MAYOR A 255 ES 255
	//Y TODO LO MENOR A 0 ES 0. AHORA SI VARIO LA PENDIENTE, ES DECIR EL 'A',
	//EL GRAFICO DEL MAPEO ME VA A CAMBIAR LA RECTA DE ESTAR MAS INCLINADA O NO.
	//PARA IMPLEMENTAR EL NEGATIVO DE LA IMAGEN SIMPLEMENTE DEBO PONER EN NEGATIVO
	//LA PENDIENTE Y EN 255 LA VARIABLE INDEPENDIENTE (-1,255);
	//SI PRODUZCO UN ESTIRAMIENTO, ES DECIR LA PENDIENTE ESTARÃ ENTRE 0 Y 1, LA
	//IMAGEN ORIGINAL SE VA OSCURECIENDO. SI COMPRIMO, ES DECIR LA PENDIENTE CADA
	//VEZ MAYOR, SE IRA VIENDO CADA VEZ MÃS BRILLOSA LA IMAGEN A PUNTO DE QUEDAR
	//COMPLETAMENTE BLANCA
	Mat LUT_Lineal(Mat img, float a, float c){
		Mat aux(1,256,CV_8UC(1));
		for (int i=0;i<256;i++){
			float s=a*i+c;
			if (s>255){
				aux.at<uchar>(i)=255;
			}
			else{
				if (s<0){
					aux.at<uchar>(i)=0;
				}
				else{
					aux.at<uchar>(i)=s;
				}
			}
		}
		Mat resultado;
		
		LUT(img,aux,resultado);
		return resultado;
	}
	
	Mat LUT_Log(Mat img, float gamma){
		Mat aux(1,256,CV_8UC(1));
		for (int i=0;i<256;i++){
			float s=(255/(log(1+255)))*log(1+i);
			if (s>255){
				aux.at<uchar>(i)=255;
			}
			else{
				if (s<0){
					aux.at<uchar>(i)=0;
				}
				else{
					aux.at<uchar>(i)=s;
				}
			}
		}
		Mat resultado;
		LUT(img,aux,resultado);
		return resultado;
	}	
	
	Mat LUT_Exp(Mat img, float gamma){
		Mat aux(1,256,CV_8UC(1));
		for (int i=0;i<256;i++){
			float s=(255/pow(255,gamma))*pow(i,gamma);
			if (s>255){
				aux.at<uchar>(i)=255;
			}
			else{
				if (s<0){
					aux.at<uchar>(i)=0;
				}
				else{
					aux.at<uchar>(i)=s;
				}
			}
		}				
		Mat resultado;
		LUT(img,aux,resultado);
		return resultado;
	}	
	
	//Contamino una imagen con ruido gaussiano
	Mat Contaminar_Gaussiano(Mat img, float media, float varianza){ 
		img.convertTo(img,CV_32F,1/255.0);
		Mat ruido(img.size(),img.type());
		randn(ruido,0,0.05);
		Mat aux=(ruido+img);
		return aux;
	}
	
	//Obtener el Error Cuadratico Medio
	float ECM(Mat img1,Mat img2){
		float error=0;
		if(img1.depth() != CV_32F || img2.depth() != CV_32F){
			cout<<endl<<"TIENE Q SER CV32F"<<endl;
		}else{
			if(img1.rows==img2.rows && img1.cols==img2.cols){
				for(int i=0;i<img1.rows;i++) { 
					for(int j=0;j<img1.cols;j++) { 
						error+=pow(img1.at<float>(i,j)-img2.at<float>(i,j),2);
					}
				}
				error=sqrt(error/(img1.rows*img1.cols));
			}else{
				cout<<endl<<"TIENEN QUE TENER EL MISMO TAMANO"<<endl;
			}}
		return error;
	}
	
	//Imagen de resolucion 8 bits y obtengo los distintos planos de cada bit
	vector<Mat> Rodaja_Planos(Mat input_img){
		Mat work_img(input_img.clone());
		vector<Mat>BitPlane;
		//Creo un string con el nombre de archivo
		string file_name("huang2_bit0.png");
		for (int i=0; i < 8; i++) {
			Mat outPut;
			//Aplico mascara AND para acceder a cada bit plane (2^i)
			bitwise_and(work_img, pow(2,i), outPut);
			//Multiplico por 255 para hacerlo blanco y negro
			outPut *= 255;
			BitPlane.push_back(outPut);
			//Escribo la imagen en el disco
			//		cv::imwrite(file_name, outPut);
			//		//Modifico el nombre del archivo
			//		file_name[10] += 1;
		}
		return BitPlane;
	}
	
	//GUIA 3 - HISTOGRAMA
	//Obtengo el histograma de una imagen
	Mat Obtener_Histograma(Mat img){
		Mat canvas=cv::Mat::zeros(200,500,CV_32F);
		Mat histo = histogram(img,255);
		normalize(histo,histo,0,1,CV_MINMAX);
		draw_graph(canvas,histo);
		return canvas;
	}
	//Obtengo el histograma ecualizado de una imagen
	Mat Histograma_Ecualizado(Mat img){
		Mat ecualizado;
		equalizeHist(img,ecualizado);
		Mat canvas2(200,500,CV_32F);
		Mat histo2 = histogram(ecualizado,255);
		normalize(histo2,histo2,0,1,CV_MINMAX);
		draw_graph(canvas2,histo2);
		return canvas2;
	}
	//Obtengo una imagen ecualizada
	Mat Imagen_Ecualizada(Mat img){
		equalizeHist(img,img);
		return img;
	}
	
	//Obtengo el kernel de un filtro pasa bajos promediador
	Mat Filtro_Promediador(int tam){
		Mat kernel(tam,tam,CV_32F);
		float aux=1/pow(tam,2);
		for (int i=0;i<kernel.rows;i++){
			for (int j=0;j<kernel.cols;j++){
				kernel.at<float>(i,j)=aux;
			}
		}
		return kernel;
	}
	//Obtengo el kernel de un filtro pasa bajos gaussiano
	Mat Filtro_Gaussiano(int tam,double sigma){
		int tamreal=tam;
		tam=tam/2;
		Mat kernel(tamreal,tamreal,CV_32F);
		// set standard deviation to 1.0
		double r, s = 2.0 * sigma * sigma;
		// sum is for normalization
		double sum = 0.0;
		// generate 3x3 kernel
		for (int x = -tam; x <= tam; x++)
		{
			for(int y = -tam; y <= tam; y++)
			{
				r = sqrt(x*x + y*y);
				kernel.at<float>(x + tam,y + tam) = (exp(-(r*r)/s))/(M_PI * s);
				sum += kernel.at<float>(x + tam,y + tam);
			}
		}
		
		// normalize the Kernel
		for(int i = 0; i < 3; ++i)
			for(int j = 0; j < 3; ++j)
			kernel.at<float>(i,j) /= sum;
		
		return kernel;
	}
	
	//Si todos los valores de la imagen son mayores que valor los pinto en blanco sino dejo en negro (pasan los mayores a un valor).
	Mat Umbral(Mat img, int valor){
		Mat aux=cv::Mat::zeros(img.size(),img.type());
		for(int i=0;i<img.rows;i++) { 
			for(int j=0;j<img.cols;j++) { 
				if (img.at<uchar>(i,j)>valor){
					aux.at<uchar>(i,j)=255;
				}
			}
		}
		return aux;
		
	}
	//Devuelvo el kernel del filtro pasa alto suma 1
	Mat Filtro_Pasa_Alto_Suma1(int tam){
		Mat kernel(tam,tam,CV_32F);
		for (int i=0;i<kernel.rows;i++){
			for (int j=0;j<kernel.cols;j++){
				kernel.at<float>(i,j)=-1;
			}
		}
		kernel.at<float>(tam/2,tam/2)=tam*tam;
		return kernel;
	}
	//Devuelvo el kernel del filtro pasa alto suma 0
	Mat Filtro_Pasa_Alto_Suma0(int tam){
		Mat kernel(tam,tam,CV_32F);
		for (int i=0;i<kernel.rows;i++){
			for (int j=0;j<kernel.cols;j++){
				kernel.at<float>(i,j)=-1;
			}
		}
		kernel.at<float>(tam/2,tam/2)=tam*tam-1;
		return kernel;
	}
	//A una imagen le aplico filtro de mascara difusa y la devuelvo
	Mat Mascara_Difusa(Mat img){
		Mat aux=img.clone();
		Mat kernel=Filtro_Promediador(5);
		Mat filtro=convolve(aux,kernel);
		Mat difusa=(aux-filtro+255)/2;
		return difusa;
		waitKey(0);
	};
	//A una imagen le aplico el filtrado de alta potencia y la devuelvo
	Mat Filtrado_Alta_Potencia(Mat img, float A){
		Mat aux=img.clone();
		Mat kernel=Filtro_Promediador(5);
		Mat filtro=convolve(aux,kernel);
		Mat alta_potencia=(A*aux-filtro+255)/2;
		return alta_potencia;
		waitKey(0);
	}
	
	//Se pasa una imagen y se dice a que tipo de pertenece, ejemplo de bandera, caricatura o personaje
	void Parecidos(Mat original){
		//Genero el histograma base para las banderas
		Mat banderas=imread("Bandera01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
		Mat histo_banderas = histogram(banderas,255);
		normalize(histo_banderas,histo_banderas,0,1,CV_MINMAX);
		
		//Genero el histograma base para las caricaturas 
		Mat caricaturas=imread("Caricaturas01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
		Mat histo_caricaturas = histogram(caricaturas,255);
		normalize(histo_caricaturas,histo_caricaturas,0,1,CV_MINMAX);
		
		//Genero el histograma base para los paisajes
		Mat paisajes=imread("Paisaje01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
		Mat histo_paisajes= histogram(paisajes,255);
		normalize(histo_paisajes,histo_paisajes,0,1,CV_MINMAX);
		
		//Genero el histograma base para los personajes
		Mat personajes=imread("Personaje01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
		Mat histo_personajes= histogram(personajes,255);
		normalize(histo_personajes,histo_personajes,0,1,CV_MINMAX);
		
		//Genero el histograma para la imagen a clasificar
		Mat histo_original= histogram(original,255);
		normalize(histo_original,histo_original,0,1,CV_MINMAX);
		
		Mat canvas_banderas(200,400,CV_32F);
		draw_graph(canvas_banderas,histo_banderas);
		
		Mat canvas_caricaturas(200,400,CV_32F);
		draw_graph(canvas_caricaturas,histo_caricaturas);
		
		Mat canvas_paisajes(200,400,CV_32F);
		draw_graph(canvas_paisajes,histo_paisajes);
		
		Mat canvas_personajes(200,400,CV_32F);
		draw_graph(canvas_personajes,histo_personajes);
		
		Mat canvas_original(200,400,CV_32F);
		draw_graph(canvas_original,histo_original);
		
		imshow("bandera",canvas_banderas);
		imshow("caricaturas",canvas_caricaturas);
		imshow("paisajes",canvas_paisajes);
		imshow("personajes",canvas_personajes);
		imshow("original",canvas_original);
		
		double cpbanderas=compareHist(histo_banderas,histo_original,CV_COMP_CORREL);
		double cpcaricaturas=compareHist(histo_caricaturas,histo_original,CV_COMP_CORREL);
		double cppaisajes=compareHist(histo_paisajes,histo_original,CV_COMP_CORREL);
		double cppersonajes=compareHist(histo_personajes,histo_original,CV_COMP_CORREL);	
		cout<<cpbanderas<<endl<<cpcaricaturas<<endl<<cppaisajes<<endl<<cppersonajes<<endl;
		
		if ((cpbanderas >= cpcaricaturas) && (cpbanderas >= cppaisajes) && (cpbanderas >= cppersonajes)){
			cout<<"La imagen ingresada pertenece al grupo de las BANDERAS"<<endl;
		}
		else{
			if ((cpcaricaturas >= cpbanderas) && (cpcaricaturas>= cppaisajes) && (cpcaricaturas>= cppersonajes)){
				cout<<"La imagen ingresada pertenece al grupo de las CARICATURAS"<<endl;
			}
			else{
				if ((cppaisajes>= cpbanderas) && (cppaisajes>= cpcaricaturas) && (cppaisajes>= cppersonajes)){
					cout<<"La imagen ingresada pertenece al grupo de los PAISAJES"<<endl;
				}
				else{
					if ((cppersonajes>= cpbanderas) && (cppersonajes>= cpcaricaturas) && (cppersonajes>= cpcaricaturas)){
						cout<<"La imagen ingresada pertenece al grupo de los PERSONAJES"<<endl;
					}
				}
			}
		}
		
		waitKey(0);
	}
	
	//GUIA 4 - COLOR
	//Obtener los colores complementarios de una imagen en HSV
	Mat Complementarios(Mat img){
		Mat img_rgb=img.clone();
		Mat img_hsv;
		cvtColor(img_rgb,img_hsv,CV_BGR2HSV);
		vector <Mat> hsv;
		split(img_hsv,hsv);
		hsv[0]=hsv[0]+90; //complemento del tono H -  Averiguar complemento
		//	hsv[1]=255-hsv[1]; //complemento de la saturacion S
		hsv[2]=255-hsv[2]; //complemento del brillo V
		//	cv::print(hsv[2]);
		merge(hsv,img_hsv);
		cvtColor(img_hsv,img_rgb,CV_HSV2BGR);
		return img_rgb;
	}
	
	//Procesamiento de imagenes en PseudoColor, asignar un color a un rango de grises de la img original
	//Le paso como parametro el B G R que quiero que pinte y el umbral que es lo menor a q valor quiero q pinte.
	//Ejemplo: Mat aux=PseudoColor(img,0,0,255,20); //pinto en rojo todo lo menor a 20
	Mat PseudoColor(Mat img, int B, int G, int R, int umbral){
		//Realizo el histograma
		Mat canvas=Obtener_Histograma(img);
		//con el histograma de la imagen, puedo ver la distribucion de grises que tiene
		//la misma. Para obtener el contenido del agua se supone que los niveles minimo
		//y maximo aproximadamente de los mismos son de 0 a 20 respectivamente.
		Mat img_color;
		cvtColor(img,img_color,COLOR_GRAY2BGR); //convierto img de gris a color para poder pintar luego en amarillo
		for (int i=0;i<img.rows;i++){
			for (int j=0;j<img.cols;j++){
				if (img.at<uchar>(i,j)<umbral){
					img_color.at<Vec3b>(i,j)[0]=B;
					img_color.at<Vec3b>(i,j)[1]=G;
					img_color.at<Vec3b>(i,j)[2]=R;
				}
			}
		}
		return img_color;
	}
	
	//Devuelvo una imagen ecualizada en RGB
	Mat Ecualizar_RGB(Mat img){
		//	Mat imgO=imread("chairs.jpg");
		//Primero genero histograma para los 3 planos de RGB y luego ecualizo.
		vector <Mat> bgr; 	
		split(img, bgr);
		equalizeHist(bgr[0],bgr[0]);
		equalizeHist(bgr[1],bgr[1]);
		equalizeHist(bgr[2],bgr[2]);
		Mat img_eqRGB;
		merge(bgr,img_eqRGB);
		return img_eqRGB;
	}	//Devuelvo una imagen ecualizada en RGB
	Mat Ecualizar_HSV(Mat img){
		//		//Ecualizo la imagen en HSV
		Mat img_HSV;
		cvtColor(img, img_HSV, CV_BGR2HSV);
		vector <Mat> hsv; 	
		split(img_HSV, hsv);
		equalizeHist(hsv[2],hsv[2]); //en estas imagenes solo basta ecualizar el canal V.
		merge(hsv,img_HSV);
		cvtColor(img_HSV, img_HSV, CV_HSV2BGR);
		return img_HSV;
	}
	
	//Funcion para hacer clicks y me devuelve los xy en un vector
	vector<Point> xy;
	int cantclick = 10;
	bool reset = false;
	
	void  Click(int event, int x, int y, int flags, void* userdata){
		if(reset){
			xy.clear();
			reset = false;
		}
		if(event == EVENT_LBUTTONDOWN){
			cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
			xy.push_back(Point(x,y));
			if(xy.size()==cantclick){
				reset = true;
				destroyWindow("My Window");
			}
		}
	}
	//Me devuelve la media de una imagen
	float Media(Mat img){
		float media=0;
		for(int i=0;i<img.rows;i++) { 
			for(int j=0;j<img.cols;j++) { 
				media+=img.at<uchar>(i,j);
			}
		}
		media/=(img.rows*img.cols);
		return media;
	}
	//Me devuelve la desviacion de una imagen
	float Desvio(Mat img, float Media){
		float desvio=0;
		for(int i=0;i<img.rows;i++) { 
			for(int j=0;j<img.cols;j++) { 
				desvio+=pow(img.at<uchar>(i,j)-Media,2);
			}
		}
		desvio/=(img.rows*img.cols);
		desvio=sqrt(desvio);
		return desvio;
	}
	
	void Segmentar_HSV(Mat img, int tam_kernel){
		//Haciendo 2 clicks obtengo los puntos para el ROI que quiero armar
		cantclick=2;
		namedWindow("My Window",CV_WINDOW_KEEPRATIO);
		imshow("My Window",img);
		setMouseCallback("My Window", Click, NULL);
		waitKey(0);
		
		//muestra representativa del color de una rosa (ROI)
		Mat roi=img(Rect(xy[0].x,xy[0].y,xy[1].x-xy[0].x,xy[1].y-xy[0].y));
		
		cvtColor(roi,roi,CV_BGR2HSV); //Convierto a HSV
		vector <Mat> hsv; 	
		split(roi, hsv);
		
		//Si muestro las imagenes del H, S y V de la img original, voy a ver que la que influye en el color de
		//las rosas es solamente el canal de H, entonces trabajo solamente en Ã©l, descartando el S y V.
		
		//Hago los histogramas para saber que region utilizar luego.
		
		Mat canvasH(200,400,CV_32F);
		Mat histoH=histogram(hsv[0],256,Mat());
		normalize(histoH,histoH,0,1,CV_MINMAX);
		draw_graph(canvasH,histoH);
		Mat canvasS(200,400,CV_32F);
		Mat histoS=histogram(hsv[1],256,Mat());
		normalize(histoS,histoS,0,1,CV_MINMAX);
		draw_graph(canvasS,histoS);
		//	stats(hsv[0]);
		
		//Realizo la umbralizacion.
		Mat mascara;
		Mat segmentacion= Mat::zeros(img.size(), img.type());    
		cvtColor(img,img,CV_BGR2HSV);
		vector <Mat> hsv2; 	
		split(img, hsv2);
		imshow("H",hsv2[0]);
		imshow("S",hsv2[1]);
		cvtColor(roi,roi,CV_HSV2BGR);
		//	Mat mean, std;
		//	meanStdDev(hsv[0], mean, std);
		float mediaH=Media(hsv[0]);
		float desvioH=Desvio(hsv[0],mediaH);
		float mediaS=Media(hsv[1]);
		float desvioS=Desvio(hsv[1],mediaH);
		cout<<"Media: "<<mediaH<<endl<<"Desvio: "<<desvioH<<endl;
		cout<<"Media: "<<mediaS<<endl<<"Desvio: "<<desvioS<<endl;
		
		//	cout<<desvio;
		inRange(img,Scalar(mediaH-desvioH,mediaS-desvioS,0),Scalar(mediaH+desvioH,mediaS+desvioS,255),mascara);
		//	inRange(img,Scalar(170,0,0),Scalar(176,255,255),mascara);
		
		//Aplico filtro a la mascara para sacar los puntitos que aparecen que no son de rosas
		Mat kernel=Filtro_Promediador(tam_kernel);
		mascara=convolve(mascara,kernel);
		for (int i=0;i<mascara.rows;i++){
			for (int j=0;j<mascara.cols;j++){
				if ((int)mascara.at<uchar>(i,j)>150){mascara.at<uchar>(i,j)=255;}
				else{mascara.at<uchar>(i,j)=0;}
			}
		}
		
		img.copyTo(segmentacion,mascara);
		cvtColor(img,img,CV_HSV2BGR);
		cvtColor(segmentacion,segmentacion,CV_HSV2BGR);
		imshow("Original",img);
		//	imshow("ROI",roi);
		namedWindow("Histograma H",CV_WINDOW_KEEPRATIO);
		imshow("Histograma H",canvasH);
		namedWindow("Histograma S",CV_WINDOW_KEEPRATIO);
		imshow("Histograma S",canvasH);
		namedWindow("Mascara",CV_WINDOW_KEEPRATIO);
		imshow("Mascara",mascara);
		namedWindow("Segmentacion HSV",CV_WINDOW_KEEPRATIO);
		imshow("Segmentacion HSV",segmentacion);
	}
	
	void TransformadaHoughCircular(Mat src){ //Imagen en color, dps convierto en gris aca
		Mat src_gray;
		/// Convert it to gray
		cvtColor( src, src_gray, CV_BGR2GRAY );
		
		/// Reduce the noise so we avoid false circle detection
		GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
		
		vector<Vec3f> circles;
		
		/// Apply the Hough Transform to find the circles
		//Parametros del HoughCircles:
		//1Âº imagen de entrada en grises
		//2Âº vector de salida de los circulos encontrados, con x,y del centro y r del radio.
		//3Âº CV_HOUGH_GRADIENT
		//4Âº 1
		//5Âº Distancia entre los centros de los circulos encontrados, si es muy pequeÃ±o varios circulos pueden ser falsamente detectados
		//si es muy grande algunos se pueden perder.
		//6Âº 
		//7Âº 
		//8Âº Minimo radio
		//9Âº Maximo radio
		
		HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
		
		/// Draw the circles detected
		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
			// circle outline
			circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}
		
		/// Show your results
		namedWindow( "Hough Circle Transform Demo", CV_WINDOW_KEEPRATIO );
		imshow( "Hough Circle Transform Demo", src );
		
		waitKey(0);
	}
	
	void HoughComun(Mat img, Mat &Gradiente, Mat &transformada, int tamaniolineas, vector <vector <Point> > &pt ){
		cvtColor(img,img,CV_BGR2GRAY);
		Canny(img,Gradiente,50,200,3); //Detecto los bordes de la imagen
		//			imshow("Original",img);
		//			imshow("Bordes",Gradiente);
		vector<Vec2f> lines;
		cvtColor(Gradiente, transformada, CV_GRAY2BGR);
		//HoughLines parÃ¡metros:
		//1Âº Salida del detector del borde.
		//2Âº Vector que almacenara los parametros ro, theta de las lineas detectadas.
		//3Âº La resolucion de ro en pixeles, usamos 1 pixel
		//4Âº La resolucion del parametro theta en radianes utilizamos un grado (CV_PI/180)
		//5Âº Umbral numero minimo de intersecciones para detectar una linea, a mayor umbral lineas mas largas.
		//6Âº y 7Âº parametros por defecto en 0
		//	HoughLines(Gradiente, lines, 1, CV_PI/180, 50, 0, 0 ); //Parametros para "letras1.tif"
		HoughLines(Gradiente, lines, 1, (CV_PI/180), tamaniolineas, 0, 0 ); //Parametros para "snowman.png"
		//			cout<<lines.size()<<endl;
		
		for( size_t i = 0; i < lines.size(); i++ )
		{
			cout<<(180*(lines[i][1])/M_PI)<<endl;
			float ro = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			vector <Point> ptt;
			double a = cos(theta), b = sin(theta);
			double x0 = a*ro, y0 = b*ro;
			pt1.x = cvRound(x0 + 1000*(-b));
			pt1.y = cvRound(y0 + 1000*(a));
			pt2.x = cvRound(x0 - 1000*(-b));
			pt2.y = cvRound(y0 - 1000*(a));
			ptt.push_back(pt1);
			ptt.push_back(pt2);
			line( transformada, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
			pt.push_back(ptt);
		}
		for(int i=0;i<lines.size();i++) { 
			
			cout<<lines[i](1)*180/M_PI<<endl;
		}
	}
	void HoughComunAngulos(Mat img, Mat &Gradiente, Mat &transformada,float angulo, int tamaniolineas, vector <vector <Point> > &pt ){
		//Le paso el angulo de las que quiero encontrar, recordar que:
		//Verticales: 0°
		//Horizontales: 90°
		
		cvtColor(img,img,CV_BGR2GRAY);
		Canny(img,Gradiente,50,200,3); //Detecto los bordes de la imagen
		vector<Vec2f> lines;
		cvtColor(Gradiente, transformada, CV_GRAY2BGR);
		HoughLines(Gradiente, lines, 1, (CV_PI/180), tamaniolineas, 0, 0 ); //Parametros para "snowman.png"
		
		for( size_t i = 0; i < lines.size(); i++ )
		{	
			cout<<(180*(lines[i][1])/M_PI)<<endl;
			if (angulo==(float)(180*(lines[i][1])/M_PI)){
				cout<<"Entro: "<<(180*(lines[i][1])/M_PI)<<endl;
				float ro = lines[i][0], theta = lines[i][1];
				Point pt1, pt2;
				vector <Point> ptt;
				double a = cos(theta), b = sin(theta);
				double x0 = a*ro, y0 = b*ro;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
				ptt.push_back(pt1);
				ptt.push_back(pt2);
				line( transformada, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
				pt.push_back(ptt);
			}
			
		}
	}
	
	void RecorrerVectorVectoresPoint(vector <vector <Point> > pt){
		for(int i=0;i<pt.size();i++) { 
			vector <Point> aux;
			aux=pt[i];
			cout<<"LINEA "<<i;
			Point p1=aux[0];
			Point p2=aux[1];
			cout<<" PUNTO 1: X: "<<p1.x<<" Y: "<<p1.y<<" - PUNTO 2: X: "<<p2.x<<" Y: "<<p2.y<<endl;
		}
	}
	
	
	
};

#endif
