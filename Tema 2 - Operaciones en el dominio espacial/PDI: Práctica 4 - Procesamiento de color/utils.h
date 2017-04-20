#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

//Funcion para mostrar varias imagenes en una ventana
//Parametros: Array con las imagenes, cantidad de las mismas, cantidad de imagenes por fila,
//ancho y alto al cual se van a llevar todas las imagenes, espaciado entre cada una
void ShowMultipleImages(Mat *images, int qty, int rowsqty, float height, float width, float spacing)
{
	Mat *images_sized = new Mat[qty];
	Size size(height,width);
	int dstWidth,dstHeight,colsqty,currentheight,currentimage,currentwidth;
	for(int i=0;i<qty;i++) { resize(images[i],images_sized[i],size); }
	
	dstWidth = qty<rowsqty? width*qty + spacing*(qty-1) : width*rowsqty + spacing*(rowsqty-1);
	dstHeight = height*(int(qty%rowsqty?qty/rowsqty+1:qty/rowsqty)) + spacing*(int(qty%rowsqty?qty/rowsqty:qty/rowsqty-1));
	Mat dst(dstHeight,dstWidth,CV_8UC(3));
	colsqty = int(qty/rowsqty)+1;
	currentheight = 0;
	currentwidth = 0;
	for(int i=0;i<colsqty;i++) { 
		for(int j=0;j<rowsqty;j++) { 
			currentimage = (i*rowsqty)+j;
			if (currentimage>=qty) break;
			currentwidth += (j?width:0) + (j?spacing:0);
			images_sized[currentimage].copyTo(dst.rowRange(currentheight, currentheight+height).colRange(currentwidth, currentwidth+width));
		}
		if (currentimage>=qty) break;
		currentheight+=height+spacing;
		currentwidth=0;
	}
	
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", dst);
	waitKey(0);
}

//int main(int argc, char** argv) {
//	
//	Mat image = imread(argv[1],1);
//	Mat *images = new Mat[10];
//	for(int i=0;i<10;i++) { images[i]=image; }
//	ShowMultipleImages(images, 10, 6, 200, 200, 10); //Verificar que si no es cuadrada no funciona
//	
//	return 0;
//} 


