#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;
	
	previous_timestamp_ = 0;
	
	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);
	
	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
	            0, 0.0225;
	
	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
	            0, 0.0009, 0,
	            0, 0, 0.09;

	//assuming acceleration noise power is 9
	noise_ax = 9;	
	noise_ay = 9;

	// init state
	VectorXd x_in = VectorXd(4);

	// state transition matrix
	MatrixXd F_in = MatrixXd(4, 4);
	F_in << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;
	
	// prediction error matrix - some uncertainly about px, py and larger uncertainly on vx, vy
	MatrixXd P_in = MatrixXd(4, 4);
	P_in << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 100, 0,
		0, 0, 0, 100;
	
	// process noise matrix, a function of time step and target's acceleration uncertainly
	MatrixXd Q_in = MatrixXd(4, 4);
	ekf_.Init(x_in, P_in, F_in, H_laser_, R_laser_, Q_in);

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		/**
		  * Initialize the state ekf_.x_ with the first measurement.
		  * Create the covariance matrix.
		*/
		// first measurement
		cout << "EKF: " << endl;
		ekf_.x_ << 1, 1, 0, 0;
	
		// compute elapsed time step
		previous_timestamp_ = measurement_pack.timestamp_;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			// need to find the Jacobian matrix for the radar sensor 
			ekf_.x_(0) = measurement_pack.raw_measurements_(0)*cos(measurement_pack.raw_measurements_(1));
			ekf_.x_(1) = measurement_pack.raw_measurements_(0)*sin(measurement_pack.raw_measurements_(1));
			ekf_.H_ = MatrixXd(3,4);
			ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
			ekf_.R_ = R_radar_;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			// for the laser sensor, no Jacobian calculation is needed
			ekf_.x_(0) = measurement_pack.raw_measurements_(0);
			ekf_.x_(1) = measurement_pack.raw_measurements_(1);
			ekf_.H_ = MatrixXd(2,4);
			ekf_.H_ << 1, 0, 0, 0,
				   0, 1, 0, 0;
			ekf_.R_ = R_laser_;
		}
		
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}
	
	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/
	
	/**
	   * Update the state transition matrix F according to the new elapsed time.
	    - Time is measured in seconds.
	   * Update the process noise covariance matrix.
	   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	 */
	
	float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0; // convert to seconds	
	previous_timestamp_ = measurement_pack.timestamp_;
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;
	ekf_.Q_ << pow(dt, 4)*noise_ax/4, 0, pow(dt, 3)*noise_ax/2, 0,
	  	   0, pow(dt, 4)*noise_ay/4, 0, pow(dt, 3)*noise_ay/2,
	  	   pow(dt, 3)*noise_ax/2, 0, pow(dt, 2)*noise_ax, 0,
	  	   0, pow(dt, 3)*noise_ay/2, 0, pow(dt, 2)*noise_ay;		

	ekf_.Predict();
	
	/*****************************************************************************
	 *  Update
	 ****************************************************************************/
	
	/**
	   * Use the sensor type to perform the update step.
	   * Update the state and covariance matrices.
	 */
	
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
	  	// Radar updates
		ekf_.H_ = MatrixXd(3,4);
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	} else {
		ekf_.H_ = MatrixXd(2,4);
		ekf_.H_ << 1, 0, 0, 0,
			   0, 1, 0, 0;
		ekf_.R_ = R_laser_;
	  	// Laser updates
		ekf_.Update(measurement_pack.raw_measurements_);
	}
	
	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
