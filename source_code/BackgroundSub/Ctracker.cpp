#include "Ctracker.h"
#include "Detection.h"
using namespace cv;
using namespace std;

size_t CTrack::NextTrackID=0;
// ---------------------------------------------------------------------------
// Track constructor.
// The track begins from initial point (pt)
// ---------------------------------------------------------------------------
CTrack::CTrack(Point2d pt, cv::Rect _brect, float dt, float Accel_noise_mag)
{

	track_id = NextTrackID;

	NextTrackID++;
	// Every track have its own Kalman filter,
	// it user for next point position prediction.
	KF = new TKalmanFilter(pt,dt,Accel_noise_mag);
	// Here stored points coordinates, used for next position prediction.
	prediction=pt;
	skipped_frames=0;
	brect = _brect; //last bounding rectangular which contour is detected. (if last frame is predicted by Kalman, then this rect is not the last position)
	
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTrack::~CTrack()
{
	// Free resources.
	delete KF;
}


// ---------------------------------------------------------------------------
// Tracker. Manage tracks. Create, remove, update.
// ---------------------------------------------------------------------------
CTracker::CTracker(float _dt, float _Accel_noise_mag, double _dist_thres, int _maximum_allowed_skipped_frames,int _max_trace_length)
{
dt=_dt;
Accel_noise_mag=_Accel_noise_mag;
dist_thres=_dist_thres;
maximum_allowed_skipped_frames=_maximum_allowed_skipped_frames;
max_trace_length=_max_trace_length;
}



// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
//void CTracker::Update(vector<Point2d>& detections)
void CTracker::Update(vector<Detection*>& detections)
{
	// -----------------------------------
	// If there is no tracks yet, then every point begins its own track.
	// -----------------------------------
	
	if(tracks.size()==0)
	{
		// If no tracks yet
		for(int i=0;i<detections.size();i++)
		{
			CTrack* tr=new CTrack(detections[i]->centroid, detections[i]->brect, dt, Accel_noise_mag);
			tracks.push_back(tr);

		}	
	}


	int N = tracks.size();		
	int M = detections.size();	
	
	
	vector< vector<double> > Cost(N,vector<double>(M));
	vector<int> assignment; 


	double dist;
	for(int i=0;i<tracks.size();i++)
	{	
		// Point2d prediction=tracks[i]->prediction;
		// cout << prediction << endl;
		for(int j=0;j < detections.size(); j++)
		{
			Point2f diff=(tracks[i]->prediction - detections[j]->centroid);
			dist = sqrtf(diff.x*diff.x + diff.y*diff.y);
			Cost[i][j]=dist;
		}
	}
	// -----------------------------------
	// Solving assignment problem (tracks and predictions of Kalman filter)
	// -----------------------------------
	
	APS.Solve(Cost,assignment,AssignmentProblemSolver::optimal);

	// -----------------------------------
	// clean assignment from pairs with large distance
	// -----------------------------------
	// Not assigned tracks
	vector<int> not_assigned_tracks;

	for(int i=0;i<assignment.size();i++)
	{
		if(assignment[i]!=-1)
		{
			if(Cost[i][assignment[i]]>dist_thres)
			{
				assignment[i]=-1;
				// Mark unassigned tracks, and increment skipped frames counter,
				// when skipped frames counter will be larger than threshold, track will be deleted.
				not_assigned_tracks.push_back(i);
				tracks[i]->skipped_frames++; //kangli
			}
		}
		else
		{			
			// If track have no assigned detect, then increment skipped frames counter.
			tracks[i]->skipped_frames++;
		}

	}

	// -----------------------------------
	// If track didn't get detects long time, remove it.
	// -----------------------------------
	for(int i=0;i<tracks.size();i++)
	{
		if(tracks[i]->skipped_frames > maximum_allowed_skipped_frames)
		{
			delete tracks[i];
			tracks.erase(tracks.begin()+i);
			assignment.erase(assignment.begin()+i);
			i--;
		}
	}
	// -----------------------------------
	// Search for unassigned detects
	// -----------------------------------
	vector<int> not_assigned_detections;
	vector<int>::iterator it;
	for(int i=0;i<detections.size();i++)
	{
		it=find(assignment.begin(), assignment.end(), i);
		if(it==assignment.end())
		{
			not_assigned_detections.push_back(i);
		}
	}

	// -----------------------------------
	// and start new tracks for them.
	// -----------------------------------
	if(not_assigned_detections.size()!=0)
	{
		for(int i=0;i<not_assigned_detections.size();i++)
		{
			CTrack* tr=new CTrack(detections[not_assigned_detections[i]]->centroid, detections[not_assigned_detections[i]]->brect,dt,Accel_noise_mag);
			tracks.push_back(tr);
		}	
	}

	// Update Kalman Filters state
	
	for(int i=0;i<assignment.size();i++)
	{
		// If track updated less than one time, than filter state is not correct.

		tracks[i]->KF->GetPrediction();

		if(assignment[i]!=-1) // If we have assigned detect, then update using its coordinates,
		{
			tracks[i]->skipped_frames=0;
			tracks[i]->prediction=tracks[i]->KF->Update(detections[assignment[i]]->centroid,1);
			tracks[i]->brect = detections[assignment[i]]->brect; //update bounding box 
		}else				  // if not continue using predictions
		{
			tracks[i]->prediction=tracks[i]->KF->Update(Point2f(0,0),0);	
			//for Kalman predicted locations, also predict the bbox. Distance delta, current predicted location - last detected location.
			if(tracks[i]->skipped_frames <  maximum_allowed_skipped_frames) {
				int last_dected_trace_index = tracks[i]->trace.size() - 1 - tracks[i]->skipped_frames;

				if(last_dected_trace_index >= 0){
					Point2f diff=(tracks[i]->prediction - tracks[i]->trace[last_dected_trace_index] );
					Point2f predicted_tl =  Point( tracks[i]->brect.tl().x + diff.x ,  tracks[i]->brect.tl().y + diff.y );
					Point2f predicted_br = Point( tracks[i]->brect.br().x + diff.x ,  tracks[i]->brect.br().y + diff.y );
					tracks[i]->brect =  Rect(predicted_tl, predicted_br);		
				}
			}

		}
		
		if(tracks[i]->trace.size()>max_trace_length)
		{
			tracks[i]->trace.erase(tracks[i]->trace.begin(),tracks[i]->trace.end()-max_trace_length);
		}

		tracks[i]->trace.push_back(tracks[i]->prediction);
		tracks[i]->KF->LastResult=tracks[i]->prediction;
	}
	return;

}

CTracker::~CTracker(void)
{
	for(int i=0;i<tracks.size();i++)
	{
	delete tracks[i];
	}
	tracks.clear();
}