#pragma once
#include "opencv2/opencv.hpp"
#include <opencv/cv.h>
using namespace cv;
using namespace std;

class SvKalmanFilter
{
public:
	//Mat 	controlMatrix;
	Mat 	errorCovPost;
	Mat 	errorCovPre;
	Mat 	gain;
	Mat 	measurementMatrix;
	Mat 	measurementNoiseCov;
	Mat 	processNoiseCov;

	Mat 	statePre;
	Mat 	statePost;
	Mat 	transitionMatrix;
	
	

	SvKalmanFilter(int dynamParams, int measureParams, int controlParams);
	SvKalmanFilter();
	~SvKalmanFilter();


	const Mat & 	correct (const Mat &measurement);
	void 	init (int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);
	//const Mat & 	predict (const Mat &control=Mat());

   const Mat & 	predict ();
private: 
	int dynamParams;
	int measureParams;
	int controlParams;
	int type;

};