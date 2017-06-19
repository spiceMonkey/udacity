#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 X* Initializes Unscented Kalman filter
 */
UKF::UKF() {

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;
	
	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;
	
	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 3;
	
	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = M_PI/8;
	
	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;
	
	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;
	
	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;
	
	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;
	
	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;


	// set number of states
	n_x_ = 5;
	// number of augmented states
	n_aug_ = 7;
	// lambda value
	lambda_ = 3 - n_aug_;

	// initial state vector
	x_ = VectorXd(n_x_);
	
	// initial covariance matrix
	// initialize P_ to be an 5x5 identity matrix
	P_ = MatrixXd::Identity(n_x_, n_x_);

	// initialize prediction matrix
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// set weights
	weights_ = VectorXd(2 * n_aug_ + 1);
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < 2 * n_aug_ + 1; i++) {
		weights_(i) = 0.5 / (n_aug_ + lambda_);
	}

	// set measurement noise variance for lidar and radar
	// lidar
	R_laser_ = MatrixXd(2, 2);
	R_laser_ <<	std_laspx_ * std_laspx_, 0,
			0, std_laspy_ * std_laspy_;
	// radar	
	R_radar_ = MatrixXd(3, 3);
	R_radar_ <<	std_radr_ * std_radr_, 0, 0,
			0, std_radphi_ * std_radphi_, 0,
			0, 0, std_radrd_ * std_radrd_;

	// not initialized until the 1st measurement data
	is_initialized_ = false;		
	
	// set default NIS to be 0
	NIS_laser_ = 0.0;
	NIS_radar_ = 0.0;


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
	 * Initialize with the 1st measurement data
	 **/
	if (!is_initialized_) {
		/**
		  * Initialize the state x_ with the first measurement.
		*/
		
		// record current time
		time_us_ = meas_package.timestamp_;

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			x_(0) = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
			x_(1) = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
			x_(3) = meas_package.raw_measurements_(1);
			x_(4) = meas_package.raw_measurements_(2);
			double vx, vy;
			// compute the tracking object's speed that is orthogonal to the radar measurement
			vx = -meas_package.raw_measurements_(2) * cos(meas_package.raw_measurements_(1));
			vy = meas_package.raw_measurements_(2) * sin(meas_package.raw_measurements_(1));
			x_(2) = sqrt(vx * vx + vy * vy);
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_(0) = meas_package.raw_measurements_(0);
			x_(1) = meas_package.raw_measurements_(1);
			x_(2) = 0;
			x_(3) = 0;
			x_(4) = 0;
		}
		
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/**
	 * Normal predict and measurement update cycle
	 */
	float dt = (meas_package.timestamp_ - time_us_)/1000000.0; // convert to seconds	
	time_us_ = meas_package.timestamp_;

	// predict	
	Prediction(dt);
	
	// update
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	  	// Radar updates
		if(use_radar_) {
			UpdateRadar(meas_package);
			// print NIS
			//cout << "Radar NIS = " << NIS_radar_ << endl;
		}
	} else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	  	// Laser updates
		if(use_laser_) {
			UpdateLidar(meas_package);
			// print NIS
			//cout << "Lidar NIS = " << NIS_laser_ << endl;
		}
	}
	
	// print the output
	//cout << "x_ = " << x_ << endl;
	//cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	/** 
	 * Step 1: generate augmented state and covariance matrices
	 **/
	//create augmented mean vector
  	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;
	
	// augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = std_a_ * std_a_;
	P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
	// square root matrix
	MatrixXd A_aug = P_aug.llt().matrixL();
	
	// sigma points matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug.col(0) = x_aug;
	
	for(int i = 0; i < n_aug_; i++) {
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A_aug.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A_aug.col(i);
	}

	/** 
	 * Step 2: generate sigma points
	 **/

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_aug(0, i);
		double py = Xsig_aug(1, i);
     		double v  = Xsig_aug(2, i);
     		double yaw = Xsig_aug(3, i);
     		double yawd = Xsig_aug(4, i);
     		double mu_a = Xsig_aug(5, i);
     		double mu_yawdd = Xsig_aug(6, i);
	
		double px_p, py_p, v_p, yaw_p, yawd_p;

		// in case yawd is zero
		if (fabs(yawd) < 0.001) {
         		px_p = px + v * cos(yaw) * delta_t;
         		py_p = py + v * sin(yaw) * delta_t;
     		} else {
         		px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
         		py_p = py + v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
     		}
		
		// also add noise variance
		px_p = px_p + 0.5 * delta_t * delta_t * cos(yaw)*mu_a;
     		py_p = py_p + 0.5 * delta_t * delta_t * sin(yaw)*mu_a;
     		v_p = v + delta_t * mu_a;
     		yaw_p = yaw + yawd * delta_t + 0.5 * delta_t * delta_t * mu_yawdd;
     		yawd_p = yawd + delta_t * mu_yawdd;
     		
     		Xsig_pred_(0, i) = px_p;
     		Xsig_pred_(1, i) = py_p;
     		Xsig_pred_(2, i) = v_p;
     		Xsig_pred_(3, i) = yaw_p;
     		Xsig_pred_(4, i) = yawd_p;
	}
	
	/** 
	 * Step 3: predict new state mean and covariance
	 **/	
  	// predict the state mean
  	x_.fill(0.0);
  	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
  	  x_ += weights_(i) * Xsig_pred_.col(i);
  	}	
	
	// predict the covariance matrix
	P_.fill(0.0);
  	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		VectorXd x_diff	= Xsig_pred_.col(i) - x_;
		while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;	
		while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;	

		P_ += weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	 * Step 1: Generate measurement sigma points
	 **/
	//create matrix for sigma points in measurement space
	// lidar has 2 measurement data
	MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);

	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    		Zsig(0, i) = Xsig_pred_(0, i);
    		Zsig(1, i) = Xsig_pred_(1, i);
	}

  	//mean predicted measurement
  	VectorXd z_pred = VectorXd(2);

	z_pred.fill(0.0);

	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
	    z_pred += weights_(i) * Zsig.col(i);
	}

	/**
	 * Step 2: UKF Update 
	 **/
	//cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, 2);
	Tc.fill(0.0);
	//measurement covariance matrix S
	MatrixXd S = MatrixXd(2,2);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    		VectorXd z_diff = Zsig.col(i) - z_pred;
	    	S += weights_(i) * z_diff * z_diff.transpose();

    		VectorXd x_diff = Xsig_pred_.col(i) - x_;
    		//angle normalization
    		while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    		while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
    		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S += R_laser_;	

	//Kalman gain
	MatrixXd K = Tc * S.inverse();

	//measurement and prediction error
  	VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

	// update the state mean and covariance matrix
	x_ += K * z_diff;
	P_ -= K * S * K.transpose(); 

	// calculate the normalized innovation squared (NIS)
	NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
	 * Step 1: Generate measurement sigma points
	 **/
	//create matrix for sigma points in measurement space
	// radar has three measurement data
	MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    		double px = Xsig_pred_(0, i);
    		double py = Xsig_pred_(1, i);
    		double v = Xsig_pred_(2, i);
    		double yaw = Xsig_pred_(3, i);
    		double yawd = Xsig_pred_(4, i);
	
		double rho = px * px + py * py;	
		// simply returns if div-by-0 happens
		if(fabs(rho) < 0.0001) {
			return;
		}

    		Zsig(0, i) = sqrt(rho);
    		Zsig(1, i) = atan2(py, px);
    		Zsig(2, i) = (px * cos(yaw) * v + py * sin(yaw) * v) / sqrt(rho);
	}

  	//mean predicted measurement
  	VectorXd z_pred = VectorXd(3);

	z_pred.fill(0.0);

	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
	    z_pred += weights_(i) * Zsig.col(i);
	}

	/**
	 * Step 2: UKF Update 
	 **/
	//cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, 3);
	Tc.fill(0.0);
	//measurement covariance matrix S
	MatrixXd S = MatrixXd(3,3);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    		VectorXd z_diff = Zsig.col(i) - z_pred;
	    	//angle normalization
	    	while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
	    	while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
	    	S += weights_(i) * z_diff * z_diff.transpose();

    		VectorXd x_diff = Xsig_pred_.col(i) - x_;
    		//angle normalization
    		while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    		while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
    		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S += R_radar_;	

	//Kalman gain
	MatrixXd K = Tc * S.inverse();

	//measurement and prediction error
  	VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  	//angle normalization
  	while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
  	while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;	

	// update the state mean and covariance matrix
	x_ += K * z_diff;
	P_ -= K * S * K.transpose();

	// calculate the normalized innovation squared (NIS)
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
