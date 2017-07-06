#ifndef PID_H
#define PID_H

enum updateState {pState, iState, dState, idle}; // states to record which parameter to update

class PID {
public:
	/*
	* Errors
	*/
	double p_error;
	double i_error;
	double d_error;
	
	/*
	* Coefficients
	*/ 
	double Kp;
	double Ki;
	double Kd;
	
	/*
	* Parameter tuning coefficients
	*/ 
	bool en_opt; // indicate whether to enable parameter tuning
	bool up; // state indicator (param increased) for parameter tuning
	bool dn; // state indicator (param decreased) for parameter tuning
	double dp; // delta of change in p
	double di; // delta of change in i
	double dd; // delta of change in d
	double cur_error; // current square error
	double best_error; // best error
	double tol_error; // error tolerance to stop optimization
	updateState cur_state; // current optimization stage
	
	/*
	* Constructor
	*/
	PID();
	
	/*
	* Destructor.
	*/
	virtual ~PID();
	
	/*
	* Initialize PID.
	*/
	void Init(double Kp, double Ki, double Kd, bool en_opt, double tol_error, double dp, double di, double dd);
	
	/*
	* Update the PID error variables given cross track error.
	*/
	void UpdateError(double cte);
	
	/*
	* Calculate the total PID error.
	*/
	double TotalError();
};

#endif /* PID_H */
