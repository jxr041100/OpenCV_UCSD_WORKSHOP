#pragma once
#include "kalmanWrapper.h"
#include "hungarianAlg.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "Detection.h"
using namespace cv;
using namespace std;

class CTrack
{
public:

	
	vector<Point2d> trace;
	static size_t NextTrackID;
	size_t track_id;
	size_t skipped_frames; 
	int id;
	Point2d prediction;

	TKalmanFilter* KF;

	cv::Rect brect; //last bounding rectangular which contour is detected. (if last frame is predicted by Kalman, then this rect is not the last position)

	CTrack(Point2d p, cv::Rect _brect, float dt, float Accel_noise_mag);

	~CTrack();


   
};


class CTracker
{
public:
	

	float dt; 
	float Accel_noise_mag;
	double dist_thres;
	int maximum_allowed_skipped_frames;
	int max_trace_length;

	vector<CTrack*> tracks;
	//void Update(vector<Point2d>& detections);
	void Update(vector<Detection*>& detections);
	CTracker(float _dt, float _Accel_noise_mag, double _dist_thres=60, int _maximum_allowed_skipped_frames=10,int _max_trace_length=10);
	~CTracker(void);
private: 
	AssignmentProblemSolver APS;
};