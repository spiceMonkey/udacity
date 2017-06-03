#include "kalman_filter.h"
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() {
	/**
	  * predict the state
	*/
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;

	return;
}

void KalmanFilter::Update(const VectorXd &z) {
	/**
	  * update the state by using Kalman Filter equations
	*/
	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

	return;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	/**
	  * update the state by using Extended Kalman Filter equations
	*/
	
    	VectorXd polar_x_(3);
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);
	
	float rho = px*px+py*py;	
	float rho_sqrt = sqrt(rho);	

	// for now simply returns if div-by-0 happens
	if(fabs(rho) < 0.0001) {
		cout << "Divided by zero happens during EKF update." << endl;
		return;
	}
	
	polar_x_(0) = rho_sqrt;
	polar_x_(1) = atan2(py, px);	
	polar_x_(2) = (px*vx+py*vy)/rho_sqrt;

	VectorXd y = z - polar_x_;
	
	//make sure the angle value is within -pi and +pi
	if(y(1)>M_PI) y(1) -= 2*M_PI;
	else if(y(1)<-M_PI) y(1) += 2*M_PI;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

	return;
	
}
