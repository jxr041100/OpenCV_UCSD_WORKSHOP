// Note: this is not a full-fledged Kalman filter
//This class try to provide same APIs as opencv KalmanFilter, However, those APIs may not be identical.
/*
Predict:

X = F*X + H*U Rolls state (X) forward to new time.

P = F*P*F^T + Q Rolls the uncertainty forward in time.

Update:

Y = M – H*X Called the innovation, = measurement – state.

S = H*P*H^T + R S= Residual covariance transformed by H+R

K = P * H^T *S^-1 K = Kalman gain = variance / residual covariance.

X = X + K*Y Update with gain the new measurement

P = (I – K * H) * P Update covariance to this time.

Where:

X = State

F = rolls X forward, typically be some time delta.

U = adds in values per unit time dt.

P = Covariance – how each thing varies compared to each other.

Y = Residual (delta of measured and last state).

M = Measurement

S = Residual of covariance.

R = Minimal innovative covariance, keeps filter from locking in to a solution.

K = Kalman gain

Q = minimal update covariance of P, keeps P from getting too small.

H = Rolls actual to predicted.

I = identity matrix.
*/

#include "SvKalmanFilter.h"
using namespace cv;
using namespace std;

SvKalmanFilter::SvKalmanFilter()
{

}

SvKalmanFilter::SvKalmanFilter(int dynamParams, int measureParams, int controlParams)
{
	init(dynamParams, measureParams, controlParams);
}


SvKalmanFilter::~SvKalmanFilter()
{

}

void 	SvKalmanFilter::init (int _dynamParams, int _measureParams, int _controlParams, int _type)
{
	dynamParams = _dynamParams;
	measureParams = _measureParams;
	controlParams = _controlParams;
	type = _type;

	//TODO, now controlParams is always 0. to implement, now we only use 4,2,0
	//X = F*X + H*U
	statePre = Mat::zeros(dynamParams, 1, type);  //state vector X
    statePost = Mat::zeros(dynamParams, 1, type);  //state vector X
    transitionMatrix = Mat::eye(dynamParams, dynamParams, type); //F transition Matrix
	processNoiseCov  = Mat::zeros(dynamParams, dynamParams, type); //P = Covariance – how each thing varies compared to each other.

	measurementMatrix = Mat::zeros(measureParams, dynamParams, type); //H , Z =HX+v
    measurementNoiseCov = Mat::eye(measureParams, measureParams, type); //R

	errorCovPre = Mat::zeros(dynamParams, dynamParams, type);
    errorCovPost = Mat::zeros(dynamParams, dynamParams, type);
	//K = P * H^T *S^-1 K = Kalman gain = variance / residual covariance.
    gain = Mat::zeros(dynamParams, measureParams, type);

}



const Mat & SvKalmanFilter::correct (const Mat &measurement)
{
	// 	S = H*P*H^T + R
	Mat ResidualCov = measurementMatrix * errorCovPre * measurementMatrix.t() + measurementNoiseCov;

	//K = P * H^T *S^-1 , kalman gain, variance / residual covariance.
	gain = (measurementMatrix * errorCovPre).t() * (ResidualCov.inv()).t();

    //Y = M – H*X, Called the innovation, = measurement – state.
    Mat residual = measurement - measurementMatrix * statePre;

	//X = X + K*Y Update with gain the new measurement
    statePost = statePre + gain*residual;

	//P = (I – K * H) * P Update covariance to this time.
    errorCovPost = errorCovPre - gain * measurementMatrix * errorCovPre;

    return statePost;
}



const Mat & SvKalmanFilter::predict ()
{
	//X = F*X  
	statePre = transitionMatrix*statePost;

   //P = F*P*F^T + Q, Rolls the uncertainty forward in time.
	errorCovPre = transitionMatrix * errorCovPost * transitionMatrix.t() + processNoiseCov;

    //for next iteration
    statePre.copyTo(statePost);
    errorCovPre.copyTo(errorCovPost);

	return statePre;
}