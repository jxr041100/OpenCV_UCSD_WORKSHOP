#pragma once
#include "kalmanWrapper.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "SvKalmanFilter.h"

#define TESTING_KALMAN
#define KALMAN_UNIT_TEST_TOLERANCE 0.000001



using namespace cv;
using namespace std;
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point2f pt,float dt,float Accel_noise_mag)
{
	//time increment (lower values makes target more "massive")
	deltatime = dt; //0.2

	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object. 
	// Process noise. (standard deviation of acceleration: ì/ñ^2)
	// shows, woh much target can accelerate.
	//float Accel_noise_mag = 0.5; 

	//4 state variables, 2 measurements
	//kalman = new KalmanFilter( 4, 2, 0 );  //openCV Kalman Filter
	kalman = new SvKalmanFilter( 4, 2, 0 );  //self Kalman Filter
	secondKalman = new KalmanFilter(4 , 2, 0);

	// Transition matrix
	kalman->transitionMatrix = (Mat_<float>(4, 4) << 1,0,deltatime,0,   0,1,0,deltatime,  0,0,1,0,  0,0,0,1);
	secondKalman->transitionMatrix = (Mat_<float>(4, 4) << 1,0,deltatime,0,   0,1,0,deltatime,  0,0,1,0,  0,0,0,1);

	// init... 
	LastResult = pt;
	kalman->statePre.at<float>(0) = pt.x; // x
	kalman->statePre.at<float>(1) = pt.y; // y

	kalman->statePre.at<float>(2) = 0;
	kalman->statePre.at<float>(3) = 0;

	kalman->statePost.at<float>(0)=pt.x;
	kalman->statePost.at<float>(1)=pt.y;

	secondKalman->statePre.at<float>(0) = pt.x; // x
	secondKalman->statePre.at<float>(1) = pt.y; // y

	secondKalman->statePre.at<float>(2) = 0;
	secondKalman->statePre.at<float>(3) = 0;

	secondKalman->statePost.at<float>(0)=pt.x;
	secondKalman->statePost.at<float>(1)=pt.y;



	setIdentity(kalman->measurementMatrix);
	setIdentity(secondKalman->measurementMatrix);

	kalman->processNoiseCov=(Mat_<float>(4, 4) << 
		pow(deltatime,4.0)/4.0	,0						,pow(deltatime,3.0)/2.0		,0,
		0						,pow(deltatime,4.0)/4.0	,0							,pow(deltatime,3.0)/2.0,
		pow(deltatime,3.0)/2.0	,0						,pow(deltatime,2.0)			,0,
		0						,pow(deltatime,3.0)/2.0	,0							,pow(deltatime,2.0));

	
	secondKalman->processNoiseCov=(Mat_<float>(4, 4) << 
		pow(deltatime,4.0)/4.0	,0						,pow(deltatime,3.0)/2.0		,0,
		0						,pow(deltatime,4.0)/4.0	,0							,pow(deltatime,3.0)/2.0,
		pow(deltatime,3.0)/2.0	,0						,pow(deltatime,2.0)			,0,
		0						,pow(deltatime,3.0)/2.0	,0							,pow(deltatime,2.0));


	kalman->processNoiseCov*=Accel_noise_mag;

	setIdentity(kalman->measurementNoiseCov, Scalar::all(0.2));

	setIdentity(kalman->errorCovPost, Scalar::all(.2));

	secondKalman->processNoiseCov*=Accel_noise_mag;

	setIdentity(secondKalman->measurementNoiseCov, Scalar::all(0.2));

	setIdentity(secondKalman->errorCovPost, Scalar::all(.2));

}
//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
	delete secondKalman;
}

//---------------------------------------------------------------------------
Point2f TKalmanFilter::GetPrediction()
{
	Mat prediction = kalman->predict();
	Mat prediction2 = secondKalman->predict();

	float delta = (prediction.at<float>(0) -  prediction2.at<float>(0)) * (prediction.at<float>(0) -  prediction2.at<float>(0))
		+  (prediction.at<float>(1)   -  prediction2.at<float>(1)) *  (prediction.at<float>(1)   -  prediction2.at<float>(1)) ;
	
	cout << "Kalman GetPrediction Deviation"<< delta <<endl;

	if (delta > KALMAN_UNIT_TEST_TOLERANCE) 
	{
		cout << "kalman filter GetPrediction testing fail! The square of the distance is "<< delta <<endl;
	}

	LastResult=Point2f(prediction.at<float>(0),prediction.at<float>(1)); 



	return LastResult;
}
//---------------------------------------------------------------------------
Point2f TKalmanFilter::Update(Point2f p, bool DataCorrect)
{
	Mat measurement(2,1,CV_32FC1);
	if(!DataCorrect)
	{
		measurement.at<float>(0) = LastResult.x;  //update using prediction
		measurement.at<float>(1) = LastResult.y;
	}
	else
	{
		measurement.at<float>(0) = p.x;  //update using measurements
		measurement.at<float>(1) = p.y;
	}
	// Correction
	Mat estimated = kalman->correct(measurement);
	Mat estimated2 = secondKalman->correct(measurement);

	LastResult.x=estimated.at<float>(0);   //update using measurements
	LastResult.y=estimated.at<float>(1);

	float delta = (estimated.at<float>(0) -  estimated2.at<float>(0)) * (estimated.at<float>(0) -  estimated2.at<float>(0))
		+  (estimated.at<float>(1)   -  estimated2.at<float>(1)) *  (estimated.at<float>(1)   -  estimated2.at<float>(1)) ;
	
	cout << "Kalman Update Deviation"<< delta <<endl;

	if (delta > KALMAN_UNIT_TEST_TOLERANCE) 
	{
		cout << "kalman filter Update testing fail! The square of the distance is "<< delta <<endl;
	}

	return LastResult;
}
//---------------------------------------------------------------------------