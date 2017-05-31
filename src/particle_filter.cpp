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
	num_particles = 100;

	default_random_engine gen;
	normal_distribution<double> x_init_dist(x, std[0]);
	normal_distribution<double> y_init_dist(y, std[1]);
	normal_distribution<double> theta_init_dist(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
			double p_x = x_init_dist(gen);
			double p_y = y_init_dist(gen);
			double p_theta = theta_init_dist(gen);
			particles.push_back(Particle{i, p_x, p_y, p_theta, 1});
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Create normal_distribution for Gaussian noise and random number generator
	//seeded with random_device to selecct amount of noise to add to each particle
  random_device rd;
  default_random_engine gen(rd());
  normal_distribution<double> x_pred_dist(0, std_pos[0]);
  normal_distribution<double> y_pred_dist(0, std_pos[1]);
  normal_distribution<double> theta_pred_dist(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i)
  {
      //predicted state values
      double px_p = particles[i].x;
      double py_p = particles[i].y;
      double theta = particles[i].theta;

      //avoid division by zero if yaw rate is ~0
      if (fabs(yaw_rate) > 0.001) {
          px_p += velocity/yaw_rate*(sin(theta + yaw_rate*delta_t) - sin(theta));
          py_p += velocity/yaw_rate*(cos(theta) - cos(theta + yaw_rate*delta_t));
      }
      else {
          px_p += velocity*delta_t*cos(theta);
          py_p += velocity*delta_t*sin(theta);
      }

      theta += yaw_rate*delta_t;

      //add noise
      px_p += x_pred_dist(gen);
      py_p += y_pred_dist(gen);
      theta += theta_pred_dist(gen);

			// Make sure theta is 0 < theta < 2*pi
			while (theta > 2*M_PI) theta -= 2.*M_PI;
			while (theta < 0) theta += 2.*M_PI;

			// Update particle states with predicted vehicle motion
      particles[i].x = px_p;
      particles[i].y = py_p;
      particles[i].theta = theta;
  }
}

vector<LandmarkObs> ParticleFilter::predictlandmarks(double sensor_range, Particle particle_pose,
		Map map_landmarks) {
			// limit landmarks to search to ones that are within sensor range
			vector<LandmarkObs> predicted_landmarks;
			for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
			{
					int id = map_landmarks.landmark_list[j].id_i;
					float landmark_x = map_landmarks.landmark_list[j].x_f;
					float landmark_y = map_landmarks.landmark_list[j].y_f;
					if (dist(landmark_x, landmark_y, particle_pose.x, particle_pose.y) < sensor_range)
					{
							predicted_landmarks.push_back(LandmarkObs{id, landmark_x, landmark_y});
					}
			}
			// Return potential landmarks within sensor range
			return predicted_landmarks;
}

vector<LandmarkObs> ParticleFilter::transformObservations(Particle const& particle_pose,
		std::vector<LandmarkObs>  obs) {
	// transform observations to map coordinate frame
	vector<LandmarkObs> transformed_obs;
	for (int j = 0; j < obs.size(); ++j)
	{
		const double x = obs[j].x;
		const double y = obs[j].y;
		double transformed_x = x*cos(particle_pose.theta) - y*sin(particle_pose.theta) + particle_pose.x;
		double transformed_y = x*sin(particle_pose.theta) + y*cos(particle_pose.theta) + particle_pose.y;
		transformed_obs.push_back(LandmarkObs{j, transformed_x, transformed_y});
	}
	// Return transformed observations
	return transformed_obs;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//Use nearest neighbor to find landmark that most liklely associates with each observation
	for (int i = 0; i < observations.size(); ++i)
	{
			double min_dist = INFINITY;
			for (int j = 0; j < predicted.size(); ++j)
			{
					double d = dist(predicted[j].x, predicted[j].y,
													observations[i].x, observations[i].y);
					if (d < min_dist)
					{
							min_dist = d;
							observations[i].id = j;
					}
			}
	}
}

double ParticleFilter::gaussianProb(double ox, double oy, double px, double py,
// Calculate Gaussian Probability of an observation given a known landmark
	double std_x, double std_y) {
	double dx = ox - px;
	double dy = oy - py;
	double var_x = std_x * std_x;
	double var_y = std_y * std_y;
	double c = (1.0 / (2 * M_PI * std_x * std_y));
	double prob = c * exp(-(dx * dx / (2 * var_x) + dy * dy / (2 * var_y)));
	return prob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution.
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	weights.clear();

	//
	for (int i = 0; i < num_particles; ++i) {
			const double particle_x = particles[i].x;
			const double particle_y = particles[i].y;
			const double particle_theta = particles[i].theta;

			// gather all landmarks within sensor range
			vector<LandmarkObs> predicted_landmarks = predictlandmarks(sensor_range, particles[i],
					map_landmarks);

			// transform observations into map coordinate frame
			vector<LandmarkObs> transformed_observations = transformObservations(particles[i],
					observations);

			// associate predictions with transformed observations using nearest neighbour
			dataAssociation(predicted_landmarks, transformed_observations);

			// update particle weight using multivariate gaussian
			double new_weight = 1.0;
			int ii = 0;
			for (auto obs: transformed_observations)
			{
					ii += 1;
					cout << " obs " << ii << " is associated with landmark" << obs.id << endl;
					LandmarkObs pred = predicted_landmarks[obs.id];
					new_weight *= gaussianProb(obs.x,obs.y,pred.x,pred.y,std_x,std_y);
			}
			cout << " reweighting particle " << i << " from " << particles[i].weight << " to " << new_weight << endl;
			particles[i].weight = new_weight;
			weights.push_back(new_weight);
		}
}

void ParticleFilter::resample() {
	default_random_engine gen;
	// Generate discrete distribution based on each partical's weight
	discrete_distribution<> resample_distribution(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; ++i)
	{
			// Select new particles based on the weighted discrete distribution
			int weighted_index = resample_distribution(gen);
			new_particles.push_back(particles[weighted_index]);
	}

	particles = new_particles;
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
