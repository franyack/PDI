#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;

int main(int argc, char** argv) {
	//create a gui window:
	namedWindow("Output",1);
//	//initialize a 120X350 matrix of black pixels:
//	Mat output = Mat::zeros( 120, 350, CV_8UC3 );
//	//write text on the matrix:
//	putText(output,
//		"Hello World :)",
//		cvPoint(15,70),
//		FONT_HERSHEY_PLAIN,
//		3,
//		cvScalar(0,255,0),
//		4);
//	//display the image:
//	imshow("Output", output);
//	//wait for the user to press any key:
//	waitKey(0);
	Mat img = imread("prueba2.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat output = img;
	//Este es solo un comentario para ver la diferencia en Git
	imshow("Output",output);
	waitKey(0);
	
	//Este es otro comentario para seguir probando Git, pero ahora usando GitHub
	
	
	
	return 0;
} 
