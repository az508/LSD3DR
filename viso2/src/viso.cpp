/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include "viso.h"

#include <math.h>

using namespace std;

VisualOdometry::VisualOdometry (parameters param) : param(param) {
  J         = 0;
  p_observe = 0;
  p_predict = 0;
  matcher   = new Matcher(param.match);
  Tr_delta  = Matrix::eye(4);
  Tr_valid  = false;
  srand(0);
}

VisualOdometry::~VisualOdometry () {
  delete matcher;
}

bool VisualOdometry::updateMotion () {
  
  // estimate motion
  vector<double> tr_delta = estimateMotion(p_matched);
  
  bool static firstTime = true;
  // on failure && not first time
  if (tr_delta.size()!=6 && !firstTime)
    return false;
  
  if(firstTime)
  {
	  firstTime = false;
	  for (int i = 0; i < 6; i++)
	  {
		  tr_delta.push_back(0);
	  }
  }
  
  
  //////////////////////
  // kalman filtering 
  // deltaT: time passed
  //////////////////////
  /*
  float deltaT = 1;
  bool success = true;
  
  // measurement
  if (success) KF_z = Matrix(6,1,tr_delta.data())/deltaT;
  else         KF_z = Matrix(6,1);
 
  // state transition matrix
  KF_A = Matrix::eye(12);
  for (int i = 0; i<12; i++)
  {
	  KF_A.val[i][i] = 0;
  }
//   for (int32_t i=0; i<6; i++)
//     KF_A.val[i][i+6] = deltaT;
 
  // observation matrix
  KF_H = Matrix(6,12);
  for (int32_t i=0; i<6; i++)
    KF_H.val[i][i] = 1;

  // process noise
  KF_Q = Matrix::eye(12);
  KF_Q.setDiag(1e-9,0,2);
  KF_Q.setDiag(1e-8,3,5);
  KF_Q.setDiag(1e-0,6,8);
  KF_Q.setDiag(1e-0,9,11);

  // measurement noise
  KF_R = Matrix::eye(6);
  KF_R.setDiag(1e-2,0,2);
  KF_R.setDiag(1e-1,3,5);
//   KF_R.setDiag(1e6,0,2);
//   KF_R.setDiag(1e5,3,5);
 
  // do not rely on measurements if estimation went wrong
  if (!success)
    KF_R = KF_R*1e6;

  // first iteration
  if (KF_x.m==0) {

    // init state x and state covariance P
    KF_x = Matrix(12,1);
    KF_x.setMat(KF_z,0,0);
    KF_P = Matrix::eye(12);

  // other iterations
  } else {

    // prediction
    KF_x = KF_A*KF_x;
    KF_P = KF_A*KF_P*(~KF_A)+KF_Q;
// for(int i = 0; i <12; i++)
// {
// 	cout<<KF_x.val[i][0]<<" ";
// }
// cout<<endl;

    // kalman gain
    Matrix K = KF_P*(~KF_H)*Matrix::inv(KF_H*KF_P*(~KF_H)+KF_R);

    // correction
    KF_x = KF_x + K*(KF_z-KF_H*KF_x);
    KF_P = KF_P - K*KF_H*KF_P;
  }
// for(int i = 0; i <12; i++)
// {
// 	cout<<KF_x.val[i][0]<<" ";
// }
// cout<<endl;
  // re-set parameter vectorx2c
//cout<<tr_delta[0]<<" "<<tr_delta[1]<<" "<<tr_delta[2]<<" "<<tr_delta[3]<<" "<<tr_delta[4]<<" "<<tr_delta[5]<<" "<<endl;
//  (KF_x*deltaT).getData(tr_delta.data(),0,0,5,0);
//cout<<tr_delta[3]<<" "<<tr_delta[4]<<" "<<tr_delta[5]<<" "<<endl;
  
  */
  // set transformation matrix (previous to current frame)
  Tr_delta = transformationVectorToMatrix(tr_delta);
  Tr_valid = true;
  
  // success
  return true;
}

Matrix VisualOdometry::transformationVectorToMatrix (vector<double> tr) {

  // extract parameters
  double rx = tr[0];
  double ry = tr[1];
  double rz = tr[2];
  double tx = tr[3];
  double ty = tr[4];
  double tz = tr[5];

  // precompute sine/cosine
  double sx = sin(rx);
  double cx = cos(rx);
  double sy = sin(ry);
  double cy = cos(ry);
  double sz = sin(rz);
  double cz = cos(rz);

  // compute transformation
  Matrix Tr(4,4);
  Tr.val[0][0] = +cy*cz;          Tr.val[0][1] = -cy*sz;          Tr.val[0][2] = +sy;    Tr.val[0][3] = tx;
  Tr.val[1][0] = +sx*sy*cz+cx*sz; Tr.val[1][1] = -sx*sy*sz+cx*cz; Tr.val[1][2] = -sx*cy; Tr.val[1][3] = ty;
  Tr.val[2][0] = -cx*sy*cz+sx*sz; Tr.val[2][1] = +cx*sy*sz+sx*cz; Tr.val[2][2] = +cx*cy; Tr.val[2][3] = tz;
  Tr.val[3][0] = 0;               Tr.val[3][1] = 0;               Tr.val[3][2] = 0;      Tr.val[3][3] = 1;
  return Tr;
}

vector<int32_t> VisualOdometry::getRandomSample(int32_t N,int32_t num) {

  // init sample and totalset
  vector<int32_t> sample;
  vector<int32_t> totalset;
  
  // create vector containing all indices
  for (int32_t i=0; i<N; i++)
    totalset.push_back(i);

  // add num indices to current sample
  sample.clear();
  for (int32_t i=0; i<num; i++) {
    int32_t j = rand()%totalset.size();
    sample.push_back(totalset[j]);
    totalset.erase(totalset.begin()+j);
  }
  
  // return sample
  return sample;
}
