/**********************************************
 * Self-Driving Car Nano-degree - Udacity
 *  Created on: December 11, 2020
 *      Author: Mathilde Badoual
 **********************************************/

#include "pid_controller.h"
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kpi, double Kii, double Kdi, double output_lim_maxi, double output_lim_mini) {
   /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   **/
   Kp = Kpi;
   Ki = Kii;
   Kd = Kdi;
   output_lim_max = output_lim_maxi;
   output_lim_min = output_lim_mini;
   dt = 0;
}


void PID::UpdateError(double cte) {
   /**
   * TODO: Update PID errors based on cte.
   **/
   _cte = cte;
   if (dt == 0) {
     _diff_cte=0.0;
   } else {
     _diff_cte = (_cte - _prev_cte) / dt;
   }
   _prev_cte = _cte;
   _int_cte += _cte * dt;
   // 
}

double PID::TotalError() {
   /**
   * TODO: Calculate and return the total error
    * The code should return a value in the interval [output_lim_mini, output_lim_maxi]
   */
    double control;
    // control = Kp*P_cte + Ki*I_cte + Kd*D_cte;
 
    control = Kp*_prev_cte + Ki*_int_cte + Kd*_diff_cte;
   //  
    if (control > output_lim_max)
    {
      control = output_lim_max;
    }
    else if (control < output_lim_min)
    {
      control = output_lim_min;
    }
    return control;
}

double PID::UpdateDeltaTime(double new_delta_time) {
   /**
   * TODO: Update the delta time with new value
   */
   dt = new_delta_time;
   return dt;
}
