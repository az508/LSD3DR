//!<
//!< OpenCV version > git clone 2014/Jun/03
//!<

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <err.h>

#include <cv.hpp> //!< OpenCV C++ class. not "cv.h" which is for C.

#include "elas.h" //!< Geiger's ACCV2010 paper implementation
#include "viso_stereo.h"

#include <boost/thread/thread.hpp>
#include <boost/graph/graph_concepts.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/core/eigen.hpp>
#include <Eigen/LU> 

//#define USE_KITTI
//#define MT_VIEWER
#define RECORD_TIME

bool update;
boost::mutex updateModelMutex;
pcl::PointCloud<pcl::PointXYZRGB> globalCloud;
int totalFileNum, currentFileNum;

class Timer
{
public:
    Timer() {timeElapsed = 0; }
    
    //start a new record
    void start() { clock_gettime(CLOCK_REALTIME, &beg_); timeElapsed = 0; }
    
    //save elapsed time and pause time record
    void pause()
	{
		clock_gettime(CLOCK_REALTIME, &end_);
		timeElapsed += end_.tv_sec - beg_.tv_sec + (end_.tv_nsec - beg_.tv_nsec) / 1000000000.;
	}
	
	//continue to record the time
	void continue_()
	{
		clock_gettime(CLOCK_REALTIME, &beg_);
	}

	double getElapsed() { return timeElapsed; }
private:
    timespec beg_, end_;
	double timeElapsed;
};

void visualize()  
{  
	// prepare visualizer named "viewer"
	boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0.5, 0.5, 0.5);
	viewer->addPointCloud<pcl::PointXYZRGB> (globalCloud.makeShared(), "my points");
	viewer->addCoordinateSystem (1.0, "global");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "my points");
	viewer->addText("Point number: n/a", 100, 100, "PointNumber");
	viewer->addText("Progress: n/a", 100, 80, "Progress");
	viewer->initCameraParameters ();

	
	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		int num = globalCloud.size();
		char text[512];
		sprintf( text, "Point number: %d", num );
		viewer->updateText(text, 100, 100, "PointNumber");
		sprintf( text, "Progress: %d/%d", currentFileNum, totalFileNum );
		viewer->updateText(text, 100, 80, "Progress");
		// Get lock on the boolean update and check if cloud was updated
		boost::mutex::scoped_lock updateLock(updateModelMutex);
		if(update)
		{
			if(!viewer->updatePointCloud(globalCloud.makeShared(), "my points"))
				//viewer->addPointCloud(globalCloud.makeShared(), "sample cloud");
			update = false;
		}
		updateLock.unlock();
		}   
} ;


    //*******************************************
    //Take homogeneous transformation matrix,
    //Output 3 by 3 rotation and 1 by 3 translation marix
    //*******************************************
void homogeneous2RT( cv::Mat& H, cv::Mat& R, cv::Mat& T)
{
	T.at<double>(0, 0) = H.at<double>(0, 3);
	T.at<double>(0, 1) = H.at<double>(1, 3);
	T.at<double>(0, 2) = H.at<double>(2, 3);
	
	R.at<double>(0, 0) = H.at<double>(0, 0);
	R.at<double>(0, 1) = H.at<double>(0, 1);
	R.at<double>(0, 2) = H.at<double>(0, 2);
	R.at<double>(1, 0) = H.at<double>(1, 0);
	R.at<double>(1, 1) = H.at<double>(1, 1);
	R.at<double>(1, 2) = H.at<double>(1, 2);
	R.at<double>(2, 0) = H.at<double>(2, 0);
	R.at<double>(2, 1) = H.at<double>(2, 1);
	R.at<double>(2, 2) = H.at<double>(2, 2);
};


    //*******************************************
    //Take 3 by 3 rotation and 1 by 3 translation marix,
    //Output homogeneous transformation matrix
    //*******************************************
void RT2homogeneous( cv::Mat& H, cv::Mat& R, cv::Mat& T)
{
	
	H.at<double>(0, 0) = R.at<double>(0, 0);	H.at<double>(0, 1) = R.at<double>(0, 1);	H.at<double>(0, 2) = R.at<double>(0, 2);	H.at<double>(0, 3) = T.at<double>(0, 0);
	H.at<double>(1, 0) = R.at<double>(1, 0);	H.at<double>(1, 1) = R.at<double>(1, 1);	H.at<double>(1, 2) = R.at<double>(1, 2);	H.at<double>(1, 3) = T.at<double>(0, 1);
	H.at<double>(2, 0) = R.at<double>(2, 0);	H.at<double>(2, 1) = R.at<double>(2, 1);	H.at<double>(2, 2) = R.at<double>(2, 2);	H.at<double>(2, 3) = T.at<double>(0, 2);
	H.at<double>(3, 0) = 0				;	H.at<double>(3, 1) = 0				;	H.at<double>(3, 2) = 0				;	H.at<double>(3, 3) = 1				;

};

float NCC(cv::Mat& keyFrame, cv::Mat& targetFrame, cv::Mat& keyPixel, cv::Mat& tarPixel,int patchSize)
{
	cv::Mat result;
	
	//*******************************************
	// use ROI to get image patch
	//*******************************************
	float u = keyPixel.at<double>(0, 0);
	float v = keyPixel.at<double>(0, 1);
	cv::Mat curPatch;
	cv::Rect curPosition(u - patchSize/2, v - patchSize /2, patchSize, patchSize);
	keyFrame(curPosition).copyTo(curPatch);
	
	
	u = tarPixel.at<double>(0, 0);
	v = tarPixel.at<double>(0, 1);
	cv::Mat tarPatch;
	cv::Rect tarPosition(u - patchSize/2, v - patchSize /2, patchSize, patchSize);
	targetFrame(tarPosition).copyTo(tarPatch);
	
	
	cv::matchTemplate(curPatch, tarPatch, result, CV_TM_CCORR_NORMED);
	return result.at<float>(patchSize/2, patchSize/2);
};


	
void loadCalibrationKITTI(std::string infile, float& baseline, float& focus, float& u0, float& v0, cv::Mat& R_cam0grayTOcam2color, cv::Mat& T_cam0grayTOcam2color, cv::Mat& R_cam2TOrectcam2, cv::Mat& R_cam0TOrectcam0, float& fb2 )
{
	// calib_cam_to_cam.txt: Camera-to-camera calibration
	// --------------------------------------------------
	// 
	//   - S_xx: 1x2 size of image xx before rectification
	//   - K_xx: 3x3 calibration matrix of camera xx before rectification
	//   - D_xx: 1x5 distortion vector of camera xx before rectification
	//   - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
	//   - T_xx: 3x1 translation vector of camera xx (extrinsic)
	//   - S_rect_xx: 1x2 size of image xx after rectification
	//   - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
	//   - P_rect_xx: 3x4 projection matrix after rectification
	// 
	// Note: When using this dataset you will most likely need to access only
	// P_rect_xx, as this matrix is valid for the rectified image sequences.
	
	// load calibration file
	fstream input(infile.c_str(), ios::in);
	if(!input.good()){
		cerr << "Could not read file: " << infile << endl;
		exit(EXIT_FAILURE);
	}
	
	std::string line;
	std::getline(input, line);//skip calib_time
	std::getline(input, line);//skip calib_dist
	//!<skip two grey scale camera
	{
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		std::getline(input, line);//skip R
		std::getline(input, line);//skip T
		
		std::getline(input, line);//skip Sr
		//read Rr
		std::string matrixName;
		input>>matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_cam0TOrectcam0.at<double>(i, j);
			}
		}
		std::getline(input, line);
		std::getline(input, line);//skip Pr
		
	}
	{
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		std::getline(input, line);//skip R
		std::getline(input, line);//skip T
		
		std::getline(input, line);//skip Sr
		std::getline(input, line);//skip Rr
		std::getline(input, line);//skip Pr
	}
	//!<get calibration of two color camera

	float fb3;
	//left color
	{
		std::string matrixName;
		float dummy;
		
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		//read R
		input>>matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_cam0grayTOcam2color.at<double>(i, j);
			}
		}
		//read T
		input >> matrixName;
		for (int i = 0; i < 3; i++)
		{
			input >> T_cam0grayTOcam2color.at<double>(0, i);
		}
		std::getline(input, line);//to next line
		std::getline(input, line);//skip Sr
		
		//read Rr
		input>>matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_cam2TOrectcam2.at<double>(i, j);
			}
		}
		
		//read Pr
		input>>matrixName;
		input>>focus;
		input>>dummy;
		input>>u0;
		input>>fb2;
		input>>dummy;
		input>>dummy;
		input>>v0;
		
		std::getline(input, line);//skip Pr
	}
	
	//right color
	float focus3;
	{
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		std::getline(input, line);//skip R
		std::getline(input, line);//skip T
		
		std::getline(input, line);//skip Sr
		std::getline(input, line);//skip Rr
		
		//read Pr
		std::string matrixName;
		float dummy;
		input>>matrixName;
		input>>focus3;
		input>>dummy;
		input>>dummy;
		input>>fb3;
		input>>dummy;
		input>>dummy;
		input>>dummy;
		
		std::getline(input, line);//skip Pr
	}
	
	baseline = fabs(fb3 / (-focus3) - fb2 / (-focus));

	input.close();
};


void loadCalibrationOpencv(float& baseline, float& focus, float& u0, float& v0, float& centerShift, cv::Mat& mx1, cv::Mat& mx2, cv::Mat& my1, cv::Mat& my2, std::vector<double>& TransformAfterRecifty)
{
	
	cv::Mat Q;
	cv::FileStorage fs;
    
	fs.open("../calibrationResult/Q.xml",   cv::FileStorage::READ);
	fs["Q"] >> Q;
	fs.release();

	fs.open("../calibrationResult/mx1.xml", cv::FileStorage::READ);
	fs["mx1"] >> mx1;
	fs.release();

	fs.open("../calibrationResult/mx2.xml", cv::FileStorage::READ);
	fs["mx2"] >> mx2;
	fs.release();

	fs.open("../calibrationResult/my1.xml", cv::FileStorage::READ);
	fs["my1"] >> my1;
	fs.release();

	fs.open("../calibrationResult/my2.xml", cv::FileStorage::READ);
	fs["my2"] >> my2;
	fs.release();
	
	
	baseline = fabs ( 1 / Q.at<double>(3,2) );
	centerShift = Q.at<double>(3,3) * baseline; //center right - center left

	focus = Q.at<double>(2,3);
	u0 = -Q.at<double>(0,3);
	v0 = -Q.at<double>(1,3);
	
	
	cv::Mat R;
	cv::Mat T;
	fs.open("../calibrationResult/R.xml", cv::FileStorage::READ);
	fs["R"] >> R;
	fs.release();
	fs.open("../calibrationResult/T.xml", cv::FileStorage::READ);
	fs["T"] >> T;
	fs.release();
	
	//!<let's push the whole transfornmation matrix into that std vector
	//!<and extract that vector in the libviso2
	
	//read and push elements from the 3x3 R and 1x3 T into the vector
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			TransformAfterRecifty.push_back( R.at<double>(i,j) );
			cout<<R.at<double>(i,j)<<" ";
		}
		TransformAfterRecifty.push_back( T.at<double>(i,0) );
		cout<<T.at<double>(i,0)<<endl;
	}
	TransformAfterRecifty.push_back( 0 );
	TransformAfterRecifty.push_back( 0 );
	TransformAfterRecifty.push_back( 0 );
	TransformAfterRecifty.push_back( 1 );
	cout<<endl;
	
}


    //*******************************************
    //Take directory name as input
    //Output a vector of images
    //*******************************************
void loadImageFromFile( std::string& dirName, std::vector<cv::Mat>& outputVector )
{
		struct dirent **namelist;
		int fileNum = scandir(dirName.c_str(), &namelist, NULL, alphasort);
		if(fileNum == -1) {
			err(EXIT_FAILURE, "%s", dirName.c_str());
		}
		
		//fileNum = 3+2;
		//(void) printf ("%d\n", fileNum);
		for (int i = 2; i < fileNum; ++i) 
		{
			//(void) printf("%s\n", namelist[i]->d_name);
			
			//std::cout<<dirName + namelist[i]->d_name<<std::endl;
			cv::Mat pic = cv::imread(dirName + namelist[i]->d_name, 1);
			outputVector.push_back(pic);
			
			free(namelist[i]);
		}
		free(namelist);
};

void estimateCameraPoseByViso2 (int imgWidth, int imgHeight, float focus, float cu, float cv, float baseline, double centerShift, 
								std::vector<cv::Mat>& LRefectedList, std::vector<cv::Mat>& RRefectedList, std::vector<cv::Mat>& cameraPoseList, 
								std::vector<double>& TransformAfterRecifty)
{		
	// set most important visual odometry parameters
	// for a full parameter list, look at: viso_stereo.h
	VisualOdometryStereo::parameters param;
	
	// calibration parameters for sequence 2010_03_09_drive_0019 
	param.calib.f  = focus; // focal length in pixels
	param.calib.cu = cu; // principal point (u-coordinate) in pixels
	param.calib.cv = cv; // principal point (v-coordinate) in pixels
	param.base     = baseline; // baseline in meters
	param.center_shift = centerShift;
	param.cu1 = cu + centerShift;
	param.cv1 = cv;
	
	
	// init visual odometry
	VisualOdometryStereo viso(param);
	viso.setTransformAfterRecifty(TransformAfterRecifty);
	
	// current pose (this matrix transforms a point from the current
	// frame's camera coordinates to the first frame's camera coordinates)
	Matrix pose = Matrix::eye(4);
	
	for (int i = 0; i < LRefectedList.size(); i++)
	{
		int32_t dims[] = {imgWidth, imgHeight, imgWidth};
		cv::Mat grayLeftRefected;
		cv::Mat grayRightRefected;
		cv::cvtColor(LRefectedList[i],grayLeftRefected,CV_BGR2GRAY);
		cv::cvtColor(RRefectedList[i],grayRightRefected,CV_BGR2GRAY);
		
		
		if (viso.process(grayLeftRefected.data,grayRightRefected.data,dims)) 
		{
		
			// on success, update current pose
			pose = pose * Matrix::inv(viso.getMotion());
		
			// output some statistics
			double num_matches = viso.getNumberOfMatches();
			double num_inliers = viso.getNumberOfInliers();
//  			cout << "Frame: " << i;
//  			cout << ", Matches: " << num_matches;
//  			cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
//  			cout << pose << endl << endl;
			
			//cout << "# " << pose.val[0][3] <<" "<<pose.val[1][3]<<" "<<pose.val[2][3] << endl;
			cout << pose.val[0][3] <<" "<<pose.val[1][3]<<" "<<pose.val[2][3] <<" "<<num_matches<<" "<<num_inliers<< endl;
			
			
			cv::Mat H = (cv::Mat_<double>(4,4) << pose.val[0][0], pose.val[0][1], pose.val[0][2], pose.val[0][3], 
												pose.val[1][0], pose.val[1][1], pose.val[1][2], pose.val[1][3], 
												pose.val[2][0], pose.val[2][1], pose.val[2][2], pose.val[2][3], 
												pose.val[3][0], pose.val[3][1], pose.val[3][2], pose.val[3][3] );
			cameraPoseList.push_back(H);
		} else {
			cout << " ... failed!" << endl;
		}
		cv::imshow( "Left",  grayLeftRefected );
		cv::imshow( "Right",  grayRightRefected );
		cv::waitKey(10);
	}
};

int main( int argc, char** argv )
{
	float baseline;
	float fb2;
	float focus;
	float u0;
	float v0;
	int imgHeight = 375;
	int imgWidth = 1242;

	std::vector<cv::Mat> rotationList;
	std::vector<cv::Mat> translationList;
	std::vector<cv::Mat> cameraPoseList;
	std::vector<cv::Mat> magicList;

	std::string dirOxts = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0027_sync/oxts/data/";
	std::string dirRightImage = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0027_sync/image_03/data/";
	std::string dirLeftImage = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/";
	
	
#ifndef USE_KITTI
	dirRightImage = "../Right/";
	dirLeftImage = "../Left/";
#endif
	
	if (argc == 3)
	{
		dirLeftImage = argv[1];
		dirRightImage = argv[2];
	}
	else
	{
		std::cout<<"Use default image path."<<endl;
	}

    //*******************************************
    //Load calibration matrix, camera position, 
    //and image pairs. Init variables.
    //*******************************************
	float centerShift = 0;
	std::vector<double> TransformAfterRecifty;
#ifdef USE_KITTI
    std::string infile = "/home/zhao/Project/KITTI/2011_09_26/calib_cam_to_cam.txt" ;
    
    //!< load camera calibration info
	cv::Mat R_cam0grayTOcam2color(3, 3, CV_64F);
	cv::Mat R_cam2TOrectcam2(3, 3, CV_64F);
	cv::Mat R_cam0TOrectcam0(3, 3, CV_64F);
	cv::Mat T_cam0grayTOcam2color(1, 3, CV_64F);
	loadCalibrationKITTI(infile, baseline, focus, u0, v0, R_cam0grayTOcam2color, T_cam0grayTOcam2color, R_cam2TOrectcam2, R_cam0TOrectcam0, fb2 );
	std::cout<<" calib_cam_to_cam.txt loaded. "<<endl;
	//std::cout<<" R_cam0grayTOcam2color: "<<endl<<R_cam0grayTOcam2color<<endl;
	//std::cout<<" R_cam2TOrectcam2: "<<endl<<R_cam2TOrectcam2<<endl;
  
	//!<get other calibration information
	//first imu to velo
	cv::Mat R_imu2velo(3, 3, CV_64F);
	cv::Mat T_imu2velo(1, 3, CV_64F);
	{
		std::string infile = "/home/zhao/Project/KITTI/2011_09_26/calib_imu_to_velo.txt" ;
		fstream input(infile.c_str(), ios::in);
		if(!input.good()){
			cerr << "Could not read file: " << infile << endl;
			exit(EXIT_FAILURE);
		}
		
		std::string line, matrixName;
		std::getline(input, line);
		
		input >> matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_imu2velo.at<double>(i, j);
			}
		}
		
		input >> matrixName;
		for (int i = 0; i < 3; i++)
		{
			input >> T_imu2velo.at<double>(0, i);
		}
		
		input.close();
	}
	std::cout<<" calib_imu_to_velo.txt loaded. "<<endl;
	//std::cout<<" R_imu2velo: "<<endl<<R_imu2velo<<endl;
	
	//then velo to camera
	cv::Mat R_velo2cam0(3, 3, CV_64F);
	cv::Mat T_velo2cam0(1, 3, CV_64F);
	{
		std::string infile = "/home/zhao/Project/KITTI/2011_09_26/calib_velo_to_cam.txt" ;
		fstream input(infile.c_str(), ios::in);
		if(!input.good()){
			cerr << "Could not read file: " << infile << endl;
			exit(EXIT_FAILURE);
		}
		
		std::string line, matrixName;
		std::getline(input, line);
		
		input >> matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_velo2cam0.at<double>(i, j);
			}
		}
		
		input >> matrixName;
		for (int i = 0; i < 3; i++)
		{
			input >> T_velo2cam0.at<double>(0, i);
		}
		
		input.close();
	}
	std::cout<<" calib_velo_to_cam.txt loaded. "<<endl;
	//std::cout<<" R_velo2cam0: "<<endl<<R_velo2cam0<<endl;
	
	TransformAfterRecifty.push_back(1); TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(-baseline);
	TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(1); TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(0);
	TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(1); TransformAfterRecifty.push_back(0);
	TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(0); TransformAfterRecifty.push_back(1);
#else
	
	cv::Mat mx1, mx2, my1, my2;
	loadCalibrationOpencv( baseline, focus, u0, v0, centerShift, mx1, mx2, my1, my2, TransformAfterRecifty);
	
#endif
	
				
				
	//!<get camera position by translate velodyne position
	/*******************************************************************
	lat:   latitude of the oxts-unit (deg)
	lon:   longitude of the oxts-unit (deg)
	alt:   altitude of the oxts-unit (m)
	roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
	pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
	yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
	vn:    velocity towards north (m/s)
	ve:    velocity towards east (m/s)
	vf:    forward velocity, i.e. parallel to earth-surface (m/s)
	vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
	vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
	ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
	ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
	ay:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
	af:    forward acceleration (m/s^2)
	al:    leftward acceleration (m/s^2)
	au:    upward acceleration (m/s^2)
	wx:    angular rate around x (rad/s)
	wy:    angular rate around y (rad/s)
	wz:    angular rate around z (rad/s)
	wf:    angular rate around forward axis (rad/s)
	wl:    angular rate around leftward axis (rad/s)
	wu:    angular rate around upward axis (rad/s)
	pos_accuracy:  velocity accuracy (north/east in m)
	vel_accuracy:  velocity accuracy (north/east in m/s)
	navstat:       navigation status (see navstat_to_string)
	numsats:       number of satellites tracked by primary GPS receiver
	posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
	velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
	orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)
	********************************************************************/
/*double lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, pos_accuracy, vel_accuracy;
	int navstat, numsats, posmode, velmode, orimode;
	{
		double mx0,my0,alt0;
		struct dirent **namelist;
		int fileNum = scandir(dirOxts.c_str(), &namelist, NULL, alphasort);
		if(fileNum == -1) {
			err(EXIT_FAILURE, "%s", dirOxts.c_str());
		}
		//(void) printf ("%d\n", fileNum);
	
		cv::Mat H_imu0(4, 4, CV_64F);
		for (int i = 2; i < fileNum; ++i) 
		{
			fstream input(dirOxts + namelist[i]->d_name, ios::in);
			if(!input.good()){
				cerr << "Could not read file: " << infile << endl;
				exit(EXIT_FAILURE);
			}
			
			input>>lat>>lon>>alt>>roll>>pitch>>yaw>>vn>>ve>>vf>>vl>>vu>>ax>>ay>>az>>af>>al>>au>>wx>>wy>>wz>>wf>>wl>>wu>>pos_accuracy>>vel_accuracy;
			input>>navstat>>numsats>>posmode>>velmode>>orimode;
			
			input.close();
			free(namelist[i]);
			
			//!<compute mercator scale from latitude
			double scale = cos(lat * M_PI / 180.0);
			
			//!< converts lat/lon coordinates to mercator coordinates using mercator scale

			const float er = 6378137; //!< Earth radius [m]
			double mx = scale * er * (lon * M_PI / 180);
			double my = scale * er * log( tan( (90+lat) * M_PI / 360) );
			//!< (mx,my): is the location of IMU/GPS in the mercator map. The world coordinate system is based on the mercator map.
			
			//!< R,T between IMU/GPS coordinate system from i-th frame to the 1st frame
			cv::Mat R_imu2imu0(3, 3, CV_64F);
			cv::Mat T_imu2imu0(1, 3, CV_64F);
			
			if(i ==2)
			{
				mx0 = mx;
				my0 = my;
				alt0 = alt;
			}
			
			//just for refine display, otherwise the pcl viewer sucks
			T_imu2imu0.at<double>(0, 0) = mx;// -mx0;
			T_imu2imu0.at<double>(0, 1) = my;// -my0;
			T_imu2imu0.at<double>(0, 2) = alt;// -alt0; //!< hight from the sea level? [m]
			//std::cout<<T_imu2imu0<<endl;
			// T = << a, b, c;
			
			//std::cout<<T_imu2imu0<<endl;
			
			//!< rotation matrix (OXTS RT3000 user manual, page 71/92)
			double rx = roll; // roll
			double ry = yaw; // pitch
			double rz = pitch; // heading 
			
			cv::Mat Rx = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx));// base => nav  (level oxts => rotated oxts)
			cv::Mat Ry = (cv::Mat_<double>(3,3) << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry));// base => nav  (level oxts => rotated oxts)
			cv::Mat Rz = (cv::Mat_<double>(3,3) << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1);// base => nav  (level oxts => rotated oxts)
			R_imu2imu0 = Rz*Ry*Rx;
			//std::cout<<rotation<<endl;
			
			cv::Mat T_imu0TOrectcam2(1, 3, CV_64F);
			cv::Mat R_imu0TOrectcam2(3, 3, CV_64F);
			//!< Now let's translate this IMUGPS position to left camera's
			{
				cv::Mat H_imu2velo(4, 4, CV_64F);
				RT2homogeneous(H_imu2velo, R_imu2velo, T_imu2velo);
				
				cv::Mat H_velo2cam0(4, 4, CV_64F);
				RT2homogeneous(H_velo2cam0, R_velo2cam0, T_velo2cam0);
				
				cv::Mat H_cam0grayTOcam2color(4, 4, CV_64F);
				RT2homogeneous(H_cam0grayTOcam2color, R_cam0grayTOcam2color, T_cam0grayTOcam2color);
				
				cv::Mat H_cam2TOrectcam2(4, 4, CV_64F);
				cv::Mat T_cam2TOrectcam2 = (cv::Mat_<double>(1,3) << 0, 0, 0);
				RT2homogeneous(H_cam2TOrectcam2, R_cam2TOrectcam2, T_cam2TOrectcam2);
				
				cv::Mat H_cam0TOrectcam0(4, 4, CV_64F);
				cv::Mat T_cam0TOrectcam0 = (cv::Mat_<double>(1,3) << 0, 0, 0);
				RT2homogeneous(H_cam0TOrectcam0, R_cam0TOrectcam0, T_cam0TOrectcam0);
				H_cam0TOrectcam0.at<double>(3, 3) = 1.00;
				
				cv::Mat H_imu2imu0(4, 4, CV_64F);
				RT2homogeneous(H_imu2imu0, R_imu2imu0, T_imu2imu0);
				if (i == 2)
				{
					H_imu2imu0.copyTo(H_imu0);
				}
				
				
				cv::Mat H_imu02imu(4, 4, CV_64F);
				
				cv::Mat H_veloToCam2(4, 4, CV_64F);
				cv::Mat H_Temp = cv::Mat::eye(4, 4, CV_64F);
				H_Temp.at<double>(3, 3) = fb2 / focus;
				H_veloToCam2 = H_Temp * H_cam0TOrectcam0 * H_velo2cam0;
				
				
				//try eigen
				Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_Eigen(H_imu2imu0.ptr<double>(), H_imu2imu0.rows, H_imu2imu0.cols);
				std::cout<<H_imu2imu0.inv()<<endl;
				
				//A_Eigen = A_Eigen.inverse();
				//cv::Mat B_OpenCV(A_Eigen.rows(), A_Eigen.cols(), CV_64FC1, A_Eigen.data());
// 				A_Eigen.transposeInPlace();
// 				Eigen::Affine3d Tran(Eigen::Matrix4d::Map(A_Eigen.data()));
// 				Tran = Tran.inverse( Eigen::Affine);
// 				std::cout<<Tran.matrix()<<endl;
// 				cv::Mat B_OpenCV(Tran.matrix().rows(), Tran.matrix().cols(), CV_64FC1, Tran.matrix().data());
// 				//cv::eigen2cv(A_Eigen, H_imu02imu);
// 				B_OpenCV = B_OpenCV.t();
// 				std::cout<<B_OpenCV<<endl;
				
				
				//<! now this matrix will trans ith frame's point(scanner frame) into world's frame(first frame's coordinate)
				
				cv::Mat H_magic(4, 4, CV_64F);
				//H_magic = H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo  * H_imu2imu0;
				H_magic = H_veloToCam2 * H_imu2velo  * H_imu2imu0;
				magicList.push_back(H_magic);
				
				cv::Mat  H_imu0TOrectcam2(4, 4, CV_64F);
				//H_imu0TOrectcam2 =   H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo  * H_imu2imu0;
				//H_imu0TOrectcam2 = H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo  * H_imu2imu0;
				//H_imu0TOrectcam2 = H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo  * ( H_imu0.inv() * H_imu2imu0  );
				//std::cout<<H_imu0.inv() * H_imu2imu0<<endl;
				

				//H_imu0TOrectcam2 = H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo * H_imu2imu0.inv();
				//H_imu0TOrectcam2 = H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo * (H_imu0.inv() * H_imu2imu0).inv();
				H_imu0TOrectcam2 = H_veloToCam2 * H_imu2velo * (H_imu0.inv() * H_imu2imu0).inv();
				
				
				//H_imu0TOrectcam2 = H_i2c * B_OpenCV;
				
				//std::cout<<H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo<<endl;
				
				
				
				homogeneous2RT(H_imu0TOrectcam2, R_imu0TOrectcam2, T_imu0TOrectcam2);
				
				//now Pc_i = H_imu0TOrectcam2 * Pw, so for Pw we need Pw = H_imu0TOrectcam2.inv() * Pc_i
				
				
				rotationList.push_back(R_imu0TOrectcam2);
				translationList.push_back(T_imu0TOrectcam2);
				
				cv::Mat Th = H_imu0.inv() * H_imu2imu0;
				cv::Mat Tt(1, 3, CV_64F);
				cv::Mat Tr(3, 3, CV_64F);
				homogeneous2RT(Th, Tr, Tt);
				//std::cout<<Tt<<endl;
			}
			//T_imu0TOrectcam2 = T_imu2imu0 + T_imu2velo + T_velo2cam0 + T_cam0grayTOcam2color;
			//R_imu0TOrectcam2 = R_cam2TOrectcam2 * R_cam0grayTOcam2color * R_velo2cam0 * R_imu2velo * R_imu2imu0;
			//rotationList.push_back(R_imu0TOrectcam2);
			//translationList.push_back(T_imu0TOrectcam2);
			//rotation = rotation * (R_imu2velo);
			
			
			//std::cout<<T_imu0TOrectcam2<<endl;
		}
		free(namelist);
	}
	std::cout<<" camera position obtained. "<<endl;*/
	
	
	//!< load image pairs
	std::vector<cv::Mat> disparityList;
	std::vector<cv::Mat> LRefectedList;
	std::vector<cv::Mat> RRefectedList;
	std::vector<cv::Mat> LNormalizedList;
	
	int fileNum;
	//!<load images obtained by left color camera
	loadImageFromFile(dirLeftImage, LRefectedList);
	std::cout<<" left color images loaded. "<<endl;
	
	//!<load images obtained by right color camera
	loadImageFromFile(dirRightImage, RRefectedList);
	std::cout<<" right color images loaded. "<<endl;
	
	//!< get the fileNum
	fileNum = LRefectedList.size();
	
	//!<get img width & height
	imgWidth = LRefectedList[0].cols;
	imgHeight = LRefectedList[0].rows;
	
#ifndef USE_KITTI
	
	for (int i = 0; i < fileNum; i++)
	{
		//!< rectifying images (and un-distorting?)
		cv::remap( LRefectedList[i], LRefectedList[i], mx1, my1, cv::INTER_LINEAR );
		cv::remap( RRefectedList[i], RRefectedList[i], mx2, my2, cv::INTER_LINEAR );
		
		
		//cv::imshow( "L", LRefectedList[i] );
		//cv::imshow( "R", RRefectedList[i] );
		//cv::waitKey(0);
		char filename[512];
		char number_str[10];
		sprintf((char*)number_str,"%08d",i);
		
		sprintf( filename, "../undistorted/left-%s.pgm", number_str );
		cv::imwrite( filename, LRefectedList[i]);
		
		sprintf( filename, "../undistorted/right-%s.pgm", number_str );
		cv::imwrite( filename, RRefectedList[i]);
		
	}
	
#endif
	
	//!<estimate camera pose
	estimateCameraPoseByViso2(imgWidth, imgHeight, focus, u0, v0, baseline, centerShift, LRefectedList, RRefectedList, cameraPoseList, TransformAfterRecifty);
	rotationList.clear();
	translationList.clear();
	for (int i = 0; i <cameraPoseList.size(); i++)
	{
		cv::Mat T(1, 3, CV_64F);
		cv::Mat R(3, 3, CV_64F);
		cv::Mat H = cameraPoseList[i].inv();
		
		homogeneous2RT(H, R, T);
		
		rotationList.push_back(R);
		translationList.push_back(T);
	}
	
//#define USE_ELAS
	//!< init variables
	Elas::parameters param;
	param.postprocess_only_left = true;
	//param.support_threshold = 0.95;
	param.disp_min = -(10 + 1) * 16;
	param.disp_max = (40 + 1) * 16;
	const int32_t dims[3] = {imgWidth, imgHeight, imgWidth};
	Elas elas( param );
	
	cv::Ptr< cv::StereoSGBM > sbm = cv::createStereoSGBM( 0, 5, 11 );
	sbm->setP1( 600 );
	sbm->setP2( 2400 );
	sbm->setBlockSize( (1 + 3) * 2 + 1 );
	sbm->setMinDisparity( param.disp_min );
	sbm->setNumDisparities( param.disp_max );
	
		
	//!<init paramaters
	float Tcov = 0.5;
	float Tdist = 5.0;
	float Tphoto = 0.7;
	float patchSize = 7;
	float pointingError = 0.5;
	float matchingError = 1.0;
	
	
	//pcl::PointCloud<pcl::PointXYZRGB> globalCloud;
	//!<Put 'globalCloud' out of the main()
	//!<also, make this cloud not empty, otherwise crash
	pcl::PointXYZRGB tmp;
	tmp.x = tmp.y = tmp.z = 0;
	globalCloud.push_back(tmp);
	
	pcl::PointCloud<pcl::PointXYZRGB> currentFrameCloud;
	pcl::PointCloud<pcl::PointXYZRGB> currentFrameCloudDS;
	

	//!<camera matrix K for reproject 3D points to image
	cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
	K.at<double>(0,0) = focus;
	K.at<double>(1,1) = focus;
	K.at<double>(2,2) = 1.0;
	K.at<double>(0,2) = u0;
	K.at<double>(1,2) = v0;
	std::cout<<" K is "<<endl<<K<<endl;
	
	
	//!<check m stereo views nearby the key frame
	//!<r is the center view of those stereo view
	const int m = 3; 
	const int r = (m - 1) / 2 + 1;
	
	
	
	std::cout<<" start main loop. "<<endl;
#ifdef MT_VIEWER	
    //boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //viewer->setBackgroundColor (0, 0, 0);
    //viewer->addPointCloud<pcl::PointXYZRGB> (globalCloud.makeShared(), "my points");
    //viewer->addCoordinateSystem (1.0, "global");
    //viewer->initCameraParameters ();
	
	
	boost::thread workerThread(visualize); 
#endif	
	
#ifdef RECORD_TIME
	//add timer there to find compution cost on each part
	Timer disparityCost;
	Timer geometricCost;
	Timer photometricCost;
	Timer fuseCost;
	Timer pointcloudCost;
#endif
	
    while ( true )
    {
		//keyframe selected by minimum distance of camera motion
		//maybe I can use some simple frame skip to instead it
		static int framecnt = -1;
		framecnt++;
		//fileNum = 5;
		if (framecnt >= fileNum)
			break;
		
		//*******************************************
		//Dense stereo matching by ELAS
		//*******************************************
		cv::Mat imageDisparity(imgHeight, imgWidth, CV_32F);
		cv::Mat imageDisparity16(imgHeight, imgWidth, CV_16S);
		cv::Mat& leftRefected = LRefectedList.at(framecnt);
		cv::Mat& rightRefected = RRefectedList.at(framecnt);
		cv::Mat grayLeftRefected;
		cv::Mat grayRightRefected;
		cv::cvtColor(leftRefected,grayLeftRefected,CV_BGR2GRAY);
		cv::cvtColor(rightRefected,grayRightRefected,CV_BGR2GRAY);

#ifdef RECORD_TIME
		disparityCost.continue_();
#endif
		
#ifdef USE_ELAS
		elas.process(grayLeftRefected.data, grayRightRefected.data, (float*)imageDisparity.data,(float*)imageDisparity.data, dims);
#else	
		sbm->compute( grayLeftRefected, grayRightRefected, imageDisparity16 );
		//sbm->compute( grayRightRefected, grayLeftRefected, imageDisparity16 );
		imageDisparity16.convertTo( imageDisparity, CV_32F );
		imageDisparity = imageDisparity /16;
#endif
		
#ifdef RECORD_TIME
		disparityCost.pause();
#endif
		
		//!<Record disparity for each frame
		disparityList.push_back(imageDisparity);
		
		
		
		double minVal, maxVal;
        minMaxLoc( imageDisparity, &minVal, &maxVal );
		cv::Mat disp8U(imgHeight, imgWidth, CV_8U);
        imageDisparity.convertTo( disp8U, CV_8UC1, 255/(maxVal - minVal), 255/(maxVal - minVal) * abs(minVal) );
		
		//save disparity here for later check
		char filename[512];
		char number_str[10];
		sprintf((char*)number_str,"%08d",framecnt);
		sprintf( filename, "../disparityMap/disp-%s.pgm", number_str );
		cv::imwrite( filename, disp8U);
		
		//cv::imshow( "disparity", disp8U );
		//cv::waitKey( 10 );
		
		//*******************************************
		//claculate the Normalized zero mean and unit variance 
		//left rectified RGB image
		//*******************************************
		//!<convert color image to grey scale 
		cv::Mat gray;
		cv::Mat grayNormalized;
		cv::cvtColor(leftRefected,gray,CV_BGR2GRAY);
		cv::Scalar mean = cv::mean(gray);
		gray = gray - mean;
		
		//double minVal, maxVal;
		minMaxLoc( gray, &minVal, &maxVal );
		gray.convertTo(grayNormalized, CV_32F, 1/(maxVal - minVal), -0.5f);
		
		LNormalizedList.push_back(grayNormalized);
		
		if ((framecnt + 1) % m != 0)
			continue;
		
		//if (framecnt < 3)
		//	continue;
		
		currentFrameCloud.clear();
		
		//center of m stereo view is the keyframe
		int keyFrame = framecnt - m + r;
		std::cout<<" start keyframe at "<<keyFrame <<"."<<endl;
		cv::imshow( "Disparity",  disp8U );
		cv::imshow( "Left",  LRefectedList[framecnt - m + 1] );
		cv::waitKey(10);
		
		std::cout<<rotationList[keyFrame]<<endl;
		std::cout<<translationList[keyFrame]<<endl;
		
		
		//use a mask to avoid reconstructing exist point
		cv::Mat mask = cv::Mat::zeros(imgHeight, imgWidth, CV_8U);

		for (int i = 0; i < globalCloud.size(); i++)
		{
			cv::Mat Ui = cv::Mat::zeros(1, 3, CV_64F);
			Ui.at<double>(0,0) = globalCloud[i].x;
			Ui.at<double>(0,1) = globalCloud[i].y;
			Ui.at<double>(0,2) = globalCloud[i].z;
			
			Ui = K * (rotationList[keyFrame] * Ui.t() + translationList[keyFrame].t());
			Ui = Ui / Ui.at<double>(0,2);
			float u = Ui.at<double>(0,0);
			float v = Ui.at<double>(0,1);
			
			//check if pixel out side the image
			if (u<0 || u>imgWidth || v <0 || v> imgHeight ||isnan(u) ||isnan(v))
			{
				continue;
			}
			
			//fill mask image
			mask.at<char>(v, u) = 255;
		}
		
		//cout<<"masked number: "<<maskcnt<<"/"<<imgHeight*imgWidth<<endl;
		
		//cv::imshow( "mask", mask );
		//cv::waitKey(10);
		
		for ( int i = 0; i < leftRefected.rows; i++ )
		{
			for ( int j = 0; j < leftRefected.cols; j++ )
			{
				//!<here add check of whether point is already reconstructed
				if (mask.at<char>(i, j) != 0)
					continue;
				
#ifdef RECORD_TIME
				geometricCost.continue_();
#endif
				//*******************************************
				//Geometric check
				//*******************************************
				float d = disparityList[keyFrame].at<float>(i, j);
				if (d < param.disp_min || d > param.disp_max)
					continue;
				d = d + centerShift;
				
				//!<compute 3D point 'hi' under camera's coordinate system 
				float z = focus * (baseline / d);
				float x = z * (j - u0) / focus;
				float y = z * (i - v0) / focus;
				cv::Mat hi(1, 3, CV_64F);
				hi.at<double>(0,0) = x;
				hi.at<double>(0,1) = y;
				hi.at<double>(0,2) = z;
				
				
				//!<convert 'hi' to 'Yi', 3D point under world coordinate system
				cv::Mat Yi(1, 3, CV_64F);
				//Yi = ( rotationList[keyFrame].inv() * hi.t() + translationList[keyFrame].t() ).t();
				Yi = ( rotationList[keyFrame].inv() *  (hi.t() - translationList[keyFrame].t()) ).t();
				//std::cout<<Yi<<endl;
				
				
				//!<compute covariance Pi as equation
				//Pi = Ji * Si * Ji'
				cv::Mat Si = cv::Mat::zeros(3, 3, CV_64F);
				Si.at<double>(0,0) = pointingError * pointingError;
				Si.at<double>(1,1) = pointingError * pointingError;
				Si.at<double>(2,2) = matchingError * matchingError;
				
				cv::Mat Ji = cv::Mat::zeros(3, 3, CV_64F);
				Ji.at<double>(0,0) = baseline / d;
				Ji.at<double>(0,2) = - (j * baseline / (d*d) );
				Ji.at<double>(1,1) = baseline / d;
				Ji.at<double>(1,2) = - (i * baseline / (d*d) );
				Ji.at<double>(2,2) = - (focus * baseline / (d*d) );
				
				cv::Mat Pi = cv::Mat::zeros(3, 3, CV_64F);
				Pi = Ji * Si * Ji.t();
				float w = Pi.at<double>(0,0) + Pi.at<double>(1,1) + Pi.at<double>(2,2);
				
				//!<check if wi < Tcov, the first check 
				if (w > Tcov)
					continue;
				
				
				//reproject 3D point 'Yi' to m neighborhood frames
				std::vector<double> weightList;
				std::vector<cv::Mat> YiList;
				bool flgReprojectCheckPassed = true;
				for (int k = 1; k <= m; k++)
				{
					
					if (k == r)
					{
						YiList.push_back(Yi);//save current point's position under current frame
						weightList.push_back(w);//save point's weight(certainty) under current frame
						continue;
					}
					int currentFrame = framecnt - m + k;
					cv::Mat& rotation = rotationList.at(currentFrame);/////////////////////////////////////////////////////////currentFrame
					cv::Mat& translation = translationList.at(currentFrame);/////////////////////////////////////////////////////////currentFrame
					cv::Mat& disparity = disparityList.at(currentFrame);/////////////////////////////////////////////////////////currentFrame
					
					
					//'Yi' will be project on current frame's 'Ui' pixel
					cv::Mat Ui = cv::Mat::zeros(1, 3, CV_64F);
					//Ui = K * (rotation * (Yi - translation).t() );
					Ui = K * (rotation * Yi.t() + translation.t());
					Ui = Ui / Ui.at<double>(0,2);
					
					//Now we got the pixel's position, let's check the disparity
					float u = Ui.at<double>(0,0);
					float v = Ui.at<double>(0,1);
					
					
					//check if pixel out side the image
					if (u<0 || u>imgWidth || v <0 || v> imgHeight ||isnan(u) ||isnan(v))
					{
						flgReprojectCheckPassed = false;
						break;
					}
					
					//get disparity
					float d = disparity.at<float>(v, u);
					
					//check if disparity is valid
					if ( (d < param.disp_min || d > param.disp_max))
					{
						flgReprojectCheckPassed = false;
						break;
					}
					
					d = d + centerShift;
					
					
					//!<check if disparity is low uncertainty
					//!<Use the same method with wi's check
					//!<since the Si is already set, we need only to refill the Ji
					
					Ji.at<double>(0,0) = baseline / d;
					Ji.at<double>(0,2) = - (j * baseline / (d*d) );
					Ji.at<double>(1,1) = baseline / d;
					Ji.at<double>(1,2) = - (i * baseline / (d*d) );
					Ji.at<double>(2,2) = - (focus * baseline / (d*d) );
					
					Pi = Ji * Si * Ji.t();
					float w = Pi.at<double>(0,0) + Pi.at<double>(1,1) + Pi.at<double>(2,2);//trace of Pi
					
					if (w > Tcov)
					{
						flgReprojectCheckPassed = false;
						break;
					}
					weightList.push_back(w);//save point's weight(certainty) under current frame
					
					
					//!<recompute point's 3D coordinate with current frame's disparity
					//!<and save it to
					float z = focus * (baseline / d);
					float x = z * (j - u0) / focus;
					float y = z * (i - v0) / focus;
					cv::Mat Yk(1, 3, CV_64F);
					//Yk.at<double>(0,0) = -y;
					//Yk.at<double>(0,1) = -z;
					//Yk.at<double>(0,2) = x;
					//Yk = ( rotation.inv() * Yk.t() + translation.t() ).t();
					
					//cv::Mat magicRotation(3, 3, CV_64F);
					//cv::Mat magicTranslation(1, 3, CV_64F);
					//homogeneous2RT(magicList[currentFrame], magicRotation, magicTranslation);
					//Yk = ( magicRotation.inv() * Yk.t() + magicTranslation.t() ).t();
					
					
					Yk.at<double>(0,0) = x;
					Yk.at<double>(0,1) = y;
					Yk.at<double>(0,2) = z;
					Yk = ( rotation.inv() *  (Yk.t() - translation.t()) ).t();
					
					//check if point is stable
					//cout<<cv::norm(Yk, Yi)<<endl;
					//cout<<Yk<<endl;
					//cout<<Yi<<endl;
					if ( Tdist < cv::norm(Yk, Yi) )
					{
						flgReprojectCheckPassed = false;
						break;
					}
					
					YiList.push_back(Yk);//save current point's position under current frame
				}
#ifdef RECORD_TIME
				geometricCost.pause();
#endif
				
				if(flgReprojectCheckPassed == false)
					continue;
				
				
				//!<check 3D difference between all reconstructed 3D points is within Tdist
				/*if (framecnt > 3 )
				{
					float mindist = 99999;
					for(int j = 0; j < globalCloud.size(); j++)
					{
						float x1 = globalCloud.at(j).x;
						float y1 = globalCloud.at(j).y;
						float z1 = globalCloud.at(j).z;
						float dis = (x-x1) * (x-x1) + (y-y1) * (y-y1) + (z-z1) * (z-z1);
						if (dis < mindist)
							mindist = dis;
					}
					if (mindist > Tdist)
						continue;
				}*/
				
#ifdef RECORD_TIME
				photometricCost.continue_();
#endif
				//*******************************************
				//Photometric check
				//*******************************************
				//In the paper this is done with image's all 3 channals
				//Use the grey scale instead
				std::vector<double> NCCSList;
				std::vector<cv::Vec3b> colorList;
				float gp = 0;
				bool flgPhotometricCheckPassed = true;
				for (int k = 1; k <= m; k++)
				{
					int currentFrame = framecnt - m + k;
					cv::Mat keyPixel;
					cv::Mat tarPixel;
				
				
					//calculate 2D position in current neighbor frame
					cv::Mat& rotation = rotationList.at(currentFrame);/////////////////////////////////////////////////////////currentFrame
					cv::Mat& translation = translationList.at(currentFrame);/////////////////////////////////////////////////////////currentFrame
					
					cv::Mat Ui(1, 3, CV_64F);
					//Ui = K * (rotation * (Yi - translation).t() );
					Ui = K * (rotation * Yi.t() + translation.t() );
					Ui = Ui / Ui.at<double>(0,2);
					tarPixel = Ui;
					
					//save pixel value in target frame
					float u = tarPixel.at<double>(0, 0);
					float v = tarPixel.at<double>(0, 1);
					
					/*cv::Mat magicRotation(3, 3, CV_64F);
					cv::Mat magicTranslation(1, 3, CV_64F);
					homogeneous2RT(magicList[currentFrame], magicRotation, magicTranslation);
					Ui = K * (rotation * (Yi - translation).t() );
					Ui = Ui / Ui.at<double>(0,2);
					u = Ui.at<double>(0, 0);
					v = Ui.at<double>(0, 1);*/
					
					
					
					cv::Vec3b bgr = LRefectedList[currentFrame].at<cv::Vec3b>(v,u);/////////////////////////////////////////////////////////currentFrame
					colorList.push_back(bgr);
					if (u - patchSize/2 <0 || u + patchSize/2>imgWidth || v - patchSize/2 <0 || v + patchSize/2> imgHeight ||isnan(u) ||isnan(v))
					{
						flgPhotometricCheckPassed = false;
						break;
					}
					
					
					//calculate 2D position in key frame
					cv::Mat& _rotation = rotationList.at(keyFrame);
					cv::Mat& _translation = translationList.at(keyFrame);
					
					//Ui = K * (_rotation * (Yi - _translation).t() );
					Ui = K * (_rotation * Yi.t() + _translation.t() );
					Ui = Ui / Ui.at<double>(0,2);
					keyPixel = Ui;
					u = keyPixel.at<double>(0, 0);
					v = keyPixel.at<double>(0, 1);
					if (u - patchSize/2 <0 || u + patchSize/2>imgWidth || v - patchSize/2 <0 || v + patchSize/2> imgHeight ||isnan(u) ||isnan(v))
					{
						flgPhotometricCheckPassed = false;
						break;
					}
				
					float NCCScore = NCC(LNormalizedList.at(keyFrame), LNormalizedList.at(currentFrame), keyPixel, tarPixel, patchSize);/////////////////////////////////////////////////////////currentFrame
					gp = gp + NCCScore;
					
				}
#ifdef RECORD_TIME
				photometricCost.pause();
#endif
				if(flgPhotometricCheckPassed == false)
					continue;
				gp = gp / 3;
				if (gp > Tphoto)
					continue;
				
				
#ifdef RECORD_TIME
				fuseCost.continue_();
#endif
				//<!if the pixel passed the both test, fuse it in to point cloud
				pcl::PointXYZRGB point;
				point.x = 0;
				point.y = 0;
				point.z = 0;
				float weightSum = 0;
				
				int pb = 0;
				int pg = 0;
				int pr = 0;
				for (int i = 0; i < m; i++)
				{
					point.x += YiList[i].at<double>(0, 0) * weightList[i];
					point.y += YiList[i].at<double>(0, 1) * weightList[i];
					point.z += YiList[i].at<double>(0, 2) * weightList[i];
					weightSum = weightSum + weightList[i];
					
					pb = colorList[i].val[0] * weightList[i];
					pg = colorList[i].val[1] * weightList[i];
					pr = colorList[i].val[2] * weightList[i];
					
				}
				point.x = point.x / weightSum;
				point.y = point.y / weightSum;
				point.z = point.z / weightSum;
				pb /= weightSum;
				pg /= weightSum;
				pr /= weightSum;
				
				uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
				point.rgb = *reinterpret_cast<float*>(&rgb);
				
				currentFrameCloud.push_back(point);
#ifdef RECORD_TIME
				fuseCost.pause();
#endif
			}
		}
		
		std::cout<<"Geometric & Photometric check finished."<<endl;
		std::cout<<currentFrameCloud.size()<<" points got."<<endl;
		
#ifdef RECORD_TIME
		pointcloudCost.continue_();
#endif
		//*******************************************
		//Remove outliers
		//*******************************************
		if (currentFrameCloud.size() != 0)
		{
			//for (int i = 0; i < currentFrameCloud.size(); i++)
			{
				//something can be done by PCL, lucky
				
				//!<Use radius removal filter
				pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
				// build the filter
				outrem.setInputCloud(currentFrameCloud.makeShared());
				outrem.setRadiusSearch(0.1* 10);
				outrem.setMinNeighborsInRadius (5);
				// apply filter
				outrem.filter (currentFrameCloud);
				
				
				//!<Use voxel grid to down sample
				pcl::VoxelGrid<pcl::PointXYZRGB> sor;
				sor.setInputCloud (currentFrameCloud.makeShared());
				sor.setLeafSize (0.2f, 0.2f, 0.2f);
				sor.filter (currentFrameCloudDS);
			}
		}
		std::cout<<"down sample & outliers removed."<<endl;
		std::cout<<currentFrameCloudDS.size()<<" points remain."<<endl;
		
		//put this point to cloud
 		for (int i = 0; i < currentFrameCloudDS.size(); i++)
 		{
 			globalCloud.push_back(currentFrameCloudDS.at(i));
 		}
// 		for (int i = 0; i < currentFrameCloud.size(); i++)
// 		{
// 			globalCloud.push_back(currentFrameCloud.at(i));
// 		}
#ifdef RECORD_TIME
		pointcloudCost.pause();
#endif
		
		
		//!< save each keyframe's reconstruction result to file
		//char filename[512];
		sprintf( filename, "frame-%d.pcd", framecnt );
		pcl::io::savePCDFileASCII (filename, currentFrameCloud);
		std::cout<<"PCD outputed."<<endl;
#ifdef MT_VIEWER
		boost::mutex::scoped_lock updateLock(updateModelMutex);
		update = true;
		currentFileNum = keyFrame + 1;
		totalFileNum = fileNum;
		updateLock.unlock();
#endif
		std::cout<<globalCloud.size()<<" points total."<<endl;
	}
#ifdef MT_VIEWER
	workerThread.join();
#endif
	
#ifdef RECORD_TIME
		std::cout<<disparityCost.getElapsed()<<" seconds used in disparity map."<<endl;
		std::cout<<geometricCost.getElapsed()<<" seconds used in geometric check."<<endl;
		std::cout<<photometricCost.getElapsed()<<" seconds used in photometric check."<<endl;
		std::cout<<fuseCost.getElapsed()<<" seconds used in fuse point."<<endl;
		std::cout<<pointcloudCost.getElapsed()<<" seconds used in pointcloud filtering."<<endl;
#endif

	//*******************************************
	//Voxel grid filtering
	//*******************************************
	
	//!<Use voxel grid to down sample
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZRGB> vox;
	vox.setInputCloud (globalCloud.makeShared());
	vox.setLeafSize (0.2f, 0.2f, 0.2f);
	vox.filter (globalCloud);
	
	std::cout<<"down sample finish."<<endl;
	
	pcl::io::savePCDFileASCII ("scene.pcd", globalCloud);
	std::cout<<"PCD outputed."<<endl;
	
    return 0;
}


