#include "Detection.h"
using namespace cv;
using namespace std;

Detection::Detection(Point2d  _centroid, cv::Rect _brect)
{
	centroid = _centroid;
	brect = _brect ;
}

Detection::Detection()
{

}


Detection::~Detection()
{

}
