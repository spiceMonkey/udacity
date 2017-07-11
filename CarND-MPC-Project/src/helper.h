#ifndef HELPER_H
#define HELPER_H

#include <math.h>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Evaluate a polynomial's slope.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += i * coeffs[i] * pow(x, i-1);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(std::vector<double> xvals, std::vector<double> yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  Eigen::Map<Eigen::VectorXd> xvals_(&xvals[0], xvals.size());
  Eigen::Map<Eigen::VectorXd> yvals_(&yvals[0], yvals.size());

  for (int i = 0; i < xvals_.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals_.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals_);
  return result;
}

#endif


