/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100; // number of particles
	particles.resize(num_particles);
	
	// create normal distributions around the provided initial values and standard deviations 
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	default_random_engine gen; // random engine setup

	// initialize each particle's position and weight
	for(int i = 0; i < num_particles; i++) {
		particles[i].id 	= i;
		particles[i].x 		= dist_x(gen);
		particles[i].y 		= dist_y(gen);
		particles[i].theta 	= dist_theta(gen);
		particles[i].weight 	= 1.0f;
	}

	is_initialized = true; // indicate the particles are initialized now
	

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// create random Gaussian noise sources 
	normal_distribution<double> gn_x(0, std_pos[0]);
	normal_distribution<double> gn_y(0, std_pos[1]);
	normal_distribution<double> gn_theta(0, std_pos[2]);
	default_random_engine gen; // random engine setup
	
	double x_p; // predicted position x
	double y_p; // predicted position y
	double yaw_p; // predicted yaw

	// different cases depending on whether yaw_rate is zero or not
	if (fabs(yaw_rate) < 0.001) {
		for (int i = 0; i  < num_particles; i++) {
			particles[i].x = particles[i].x + velocity * cos(particles[i].theta) * delta_t + gn_x(gen); 
			particles[i].y = particles[i].y + velocity * sin(particles[i].theta) * delta_t + gn_y(gen); 
			particles[i].theta = particles[i].theta + yaw_rate * delta_t + gn_theta(gen);
		}
	}
	else {
		for (int i = 0; i < num_particles; i++) {
			particles[i].x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + gn_x(gen); 
			particles[i].y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + gn_y(gen); 
			particles[i].theta = particles[i].theta + yaw_rate * delta_t + gn_theta(gen);
		}

	}
	

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (int i = 0; i < num_particles; i++) {
		double x_p 	= particles[i].x;
		double y_p 	= particles[i].y;
		double theta_p 	= particles[i].theta;
		
		// convert the observation in local coordinates to global coordinates
		for (int o = 0; o < observations.size(); o++) {
			double x_ob_loc  = observations[o].x;
			double y_ob_loc  = observations[o].y;
			double x_ob_glob = x_ob_loc * cos(theta_p) - y_ob_loc * sin(theta_p) + x_p;
			double y_ob_glob = x_ob_loc * sin(theta_p) + y_ob_loc * cos(theta_p) + y_p;
			
			double dist_min = sensor_range;
			double dist_tmp;
			int lm_id = 0;

			for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
				dist_tmp = dist(x_ob_glob, y_ob_glob, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f);
				if (dist_tmp < dist_min) {
					dist_min = dist_tmp;
					lm_id = map_landmarks.landmark_list[l].id_i;
					
				}
			}
			particles[i].associations.push_back(lm_id);
		}
		
		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
