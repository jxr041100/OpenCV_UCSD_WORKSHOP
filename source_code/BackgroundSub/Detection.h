#pragma once
#include "opencv2/opencv.hpp"
#include <opencv/cv.h>
using namespace cv;
using namespace std;

class Detection
{
public:

	
	Point2d centroid;
	cv::Rect brect; //last bounding rectangular which contour is detected. (if last frame is predicted by Kalman, then this rect is not the last position)
	Detection(Point2d _centroid, cv::Rect _brect );

	Detection();

	~Detection();


   
};