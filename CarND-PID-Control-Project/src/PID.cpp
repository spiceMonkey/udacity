#include "PID.h"

using namespace std;

/*
* Completed PID class.
*/


PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd, bool en_opt, double tol_error, double dp, double di, double dd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;
	this->en_opt = en_opt;
	this->tol_error = tol_error;
	
	this -> dp = dp;
	this -> di = di;
	this -> dd = dd;

	p_error = 0;
	i_error = 0;
	d_error = 0;
	best_error = -1;
	cur_state = idle;
	up = false;
	dn = false;
}

void PID::UpdateError(double cte) {

	d_error = cte - p_error; // p_error equals the previous cte error
	i_error += cte; // i_error is the integral of past cte errors
	p_error = cte; // p_error is simply current cte

	// PID coefficients optimization procedures - use twindle method
	if(en_opt && (dp + di + dd > tol_error)) {
		cur_error = cte * cte;	
		// parameter tuning starts here	
		// it starts with p coefficients, then d and i coeffiencts
		if (cur_state == pState) { 
			if (!up) { // first try increase parameter
				Kp += dp;
				up = true;
			}
			else if (up && !dn) {
				if (cur_error < best_error) { // if resulting error is smaller
					best_error = cur_error; // set best error to be current one
					dp *= 1.1; // further increase delta
					cur_state = dState; // move to d parameter update
					up = false;
					dn = false;
				}
				else { // otherwise decrease kp
					Kp -= 2*dp;
					dn = true;
				}
			}
			else if (up && dn) { // if both increasing and decreasing parameters have been tested
				if (cur_error < best_error) { // if decreseing parameter results in a smaller error
					best_error = cur_error; // keep it as the best error
					dp *= 1.1; // further increase the delta
				}
				else {
					Kp += dp; // if neither increasing or decreasing the parameter improves error 
					dp *= 0.9; // shrink the delta size
				}
				cur_state = dState; // move to d parameter update
				up = false;
				dn = false;
			}
		
		}
		// the same procedure as p paremeter is used for the d parameter - then moves to i
		if (cur_state == dState) {
			if (!up) {
				Kd += dd;
				up = true;
			}
			else if (up && !dn) {
				if (cur_error < best_error) {
					best_error = cur_error;
					dd *= 1.1;
					cur_state = iState;
					up = false;
					dn = false;
				}
				else {
					Kd -= 2*dd;
					dn = true;
				}
			}
			else if (up && dn) {
				if (cur_error < best_error) {
					best_error = cur_error;
					dd *= 1.1;
				}
				else {
					Kd += dd;
					dd *= 0.9;
				}
				cur_state = iState;
				up = false;
				dn = false;
			}
		}
		// same happens for i, then moves back to p
		if (cur_state == iState) {
			if (!up) {
				Ki += di;
				up = true;
			}
			else if (up && !dn) {
				if (cur_error < best_error) {
					best_error = cur_error;
					di *= 1.1;
					cur_state = pState;
					up = false;
					dn = false;
				}
				else {
					Ki -= 2*di;
					dn = true;
				}
			}
			else if (up && dn) {
				if (cur_error < best_error) {
					best_error = cur_error;
					di *= 1.1;
				}
				else {
					Ki += di;
					di *= 0.9;
				}
				cur_state = pState;
				up = false;
				dn = false;
			}
		}
		// initially the state is in idle, just initialize the best error to be current error
		if (cur_state == idle) {
			best_error = cur_error;
			cur_state = pState;
			up = false;
			dn = false;
		}
	}
}

double PID::TotalError() {
	return -Kp * p_error - Ki * i_error - Kd * d_error;
}


