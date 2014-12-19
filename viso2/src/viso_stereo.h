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

#ifndef VISO_STEREO_H
#define VISO_STEREO_H

#include "viso.h"

class VisualOdometryStereo : public VisualOdometry {

public:

  // stereo-specific parameters (mandatory: base)
  struct parameters : public VisualOdometry::parameters {
    double  base;             // baseline (meters)
    int32_t ransac_iters;     // number of RANSAC iterations
    double  inlier_threshold; // fundamental matrix inlier threshold
    bool    reweighting;      // lower border weights (more robust to calibration errors)
    double  center_shift;     // from Q33 for converged disparity
    double cu1;
	double cv1;
    parameters () {
      base             = 1.0;
      ransac_iters     = 300;
      inlier_threshold = 2.0;
      reweighting      = true;
	  center_shift     = 0.0;
	  cu1 = 0;
	  cv1 = 0;
    }
  };

  // constructor, takes as inpute a parameter structure
  VisualOdometryStereo (parameters param);
  
  // deconstructor
  ~VisualOdometryStereo ();
  
  // process a new images, push the images back to an internal ring buffer.
  // valid motion estimates are available after calling process for two times.
  // inputs: I1 ........ pointer to rectified left image (uint8, row-aligned)
  //         I2 ........ pointer to rectified right image (uint8, row-aligned)
  //         dims[0] ... width of I1 and I2 (both must be of same size)
  //         dims[1] ... height of I1 and I2 (both must be of same size)
  //         dims[2] ... bytes per line (often equal to width)
  //         replace ... replace current images with I1 and I2, without copying last current
  //                     images to previous images internally. this option can be used
  //                     when small/no motions are observed to obtain Tr_delta wrt
  //                     an older coordinate system / time step than the previous one.
  // output: returns false if an error occured
  bool process (uint8_t *I1,uint8_t *I2,int32_t* dims,bool replace=false);

  using VisualOdometry::process;


  void setTransformAfterRecifty(std::vector<double>& TransformAfterRecifty)
  {
	T00 = TransformAfterRecifty[0], T01 = TransformAfterRecifty[1], T02 = TransformAfterRecifty[2], T03 = TransformAfterRecifty[3],
	T10 = TransformAfterRecifty[4], T11 = TransformAfterRecifty[5], T12 = TransformAfterRecifty[6], T13 = TransformAfterRecifty[7],
	T20 = TransformAfterRecifty[8], T21 = TransformAfterRecifty[9], T22 = TransformAfterRecifty[10], T23 = TransformAfterRecifty[11],
	T30 = TransformAfterRecifty[12], T31 = TransformAfterRecifty[13], T32 = TransformAfterRecifty[14], T33 = TransformAfterRecifty[15];
	
	matcher->setTransformAfterRecifty( TransformAfterRecifty );
	
			T00 = 1, T01 = 0, T02 = 0, T03 = -41.7890,
			T10 = 0, T11 = 1, T12 = 0, T13 = 0,
			T20 = 0, T21 = 0, T22 = 1, T23 = 0,
			T30 = 0, T31 = 0, T32 = 0, T33 = 1;
  }

private:

  std::vector<double>  estimateMotion (std::vector<Matcher::p_match> p_matched);
  enum                 result { UPDATED, FAILED, CONVERGED };  
  result               updateParameters(std::vector<Matcher::p_match> &p_matched,std::vector<int32_t> &active,std::vector<double> &tr,double step_size,double eps);
  void                 computeObservations(std::vector<Matcher::p_match> &p_matched,std::vector<int32_t> &active);
  void                 computeResidualsAndJacobian(std::vector<double> &tr,std::vector<int32_t> &active);
  std::vector<int32_t> getInlier(std::vector<Matcher::p_match> &p_matched,std::vector<double> &tr);

  double *X,*Y,*Z;    // 3d points
  double *p_residual; // residuals (p_residual=p_observe-p_predict)
  
	double 	T00 , T01 , T02 , T03 ,
			T10 , T11 , T12 , T13 ,
			T20 , T21 , T22 , T23 ,
			T30 , T31 , T32 , T33 ;
  
  // parameters
  parameters param;
};

#endif // VISO_STEREO_H

