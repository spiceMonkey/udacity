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
		particles[i].weight 	= 1.0;
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

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	float x_p, y_p, theta_p; // particle's position information
	float x_ob_loc, y_ob_loc; // variable to store measurement result in local coordinates
	float x_ob_glob, y_ob_glob; // measurement result converted to global coordinates
	float x_ob_glob_min, y_ob_glob_min; // stores the min-dist measurement results in global coordinates
	float x_lm_glob, y_lm_glob; // landmark pos in global coordinates
	float x_lm_glob_min, y_lm_glob_min; // store the closest landmark pos in global coordinates
	double weight_p; 
	float dist_min, dist_tmp; // intermediate variable used to find the min-distance between measurement and landmark

	float std_x = std_landmark[0];
	float std_y = std_landmark[1];

	vector<int> best_lm; // optional, debug purpose, best landmark
	vector<double> best_sx; // optional, debug purpose, best sense_x in global coordinates
	vector<double> best_sy; // optional, debug purpose, best sense_y in global coordinates
	int best_lm_id = 0;
		
	double weight_sum = 0;	
	for (int i = 0; i < num_particles; i++) {
		// current particle position
		x_p 	= particles[i].x; 
		y_p 	= particles[i].y;
		theta_p = particles[i].theta;
	
		// current particle weight	
		weight_p = particles[i].weight;	
		
		best_lm.clear();
		best_sx.clear();
		best_sy.clear();
		
		// convert the observations in local coordinates to global coordinates
		for (int o = 0; o < observations.size(); o++) {
			x_ob_loc  = observations[o].x;
			y_ob_loc  = observations[o].y;
			x_ob_glob = x_ob_loc * cos(theta_p) - y_ob_loc * sin(theta_p) + x_p;
			y_ob_glob = x_ob_loc * sin(theta_p) + y_ob_loc * cos(theta_p) + y_p;
			
			dist_min = sensor_range;
			dist_tmp;

			for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
				x_lm_glob = map_landmarks.landmark_list[l].x_f;
				y_lm_glob = map_landmarks.landmark_list[l].y_f;
				dist_tmp = dist(x_ob_glob, y_ob_glob, x_lm_glob, y_lm_glob);
				if (dist_tmp < dist_min) {
					dist_min = dist_tmp;
					best_lm_id = map_landmarks.landmark_list[l].id_i;
					x_ob_glob_min = x_ob_glob;
					y_ob_glob_min = y_ob_glob;
					x_lm_glob_min = x_lm_glob;
					y_lm_glob_min = y_lm_glob;
				}
			}
			if (dist_min < sensor_range) {
				// save the closest landmark and corresponding measurement in global coordinates
				best_lm.push_back(best_lm_id);
				best_sx.push_back(x_ob_glob_min);
				best_sy.push_back(y_ob_glob_min);
				// update weights
				weight_p *= 1 / (2 * M_PI * std_x * std_y) * exp(-pow((x_lm_glob_min - x_ob_glob_min) / std_x, 2.0) / 2)  * exp(-pow((y_lm_glob_min - y_ob_glob_min) / std_y, 2.0) / 2);
			}
		}
	
		weight_sum += weight_p; // need to renormalize weights to avoid clipping to 0 when they become small

		particles[i].weight = weight_p;
		particles[i] = SetAssociations(particles[i], best_lm, best_sx, best_sy);	
	}
	
	for (int i = 0; i < num_particles; i++) particles[i].weight = particles[i].weight / weight_sum; // weight normalization
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// use resampling wheel to resample the particles based on their weights 
	double weight_max = -1.0; // find the maximum weight among all the particles
	double beta = 0; // sampling beta
	
	uniform_real_distribution<> dist_samp(0, 1); // uniform distribution sampling
	default_random_engine gen; // random engine setup done
	
	int index = int(dist_samp(gen) * num_particles); // index

	vector<Particle> particle_samp(num_particles); // stores the re-sampled particles	

	// find the maximum weight first
	for (int i = 0; i < num_particles; i++) {
		if(particles[i].weight > weight_max) {
			weight_max = particles[i].weight;
		}
	}
	
	// random sampling now	
	for (int i = 0; i < num_particles; i++) {
		beta += dist_samp(gen) * 2 * weight_max;
			while(particles[index].weight < beta) {
				beta -= particles[index].weight;
				index = (index + 1) % num_particles;
			}
		particle_samp[i]=particles[index];
	}
	
	// re-assign weights to particles
	particles = particle_samp;
	
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
