import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class VTOLControl(BaseControl):
    """VTOL (Vertical Takeoff and Landing) control class.
    
    Combines hover capabilities with forward flight:
    - HOVER mode: Uses quadcopter-style control for takeoff/landing
    - CRUISE mode: Forward flight with banking turns
    - Auto-transitions between modes based on altitude and stability
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Initialize VTOL controller.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (should be DroneModel.FIXEDWING for VTOL).
        g : float, optional
            The gravitational acceleration in m/s^2.
        """
        # Flight modes
        self.HOVER = "hover"
        self.CRUISE = "cruise"
        self.mode = self.HOVER
        
        # Use CF2X controller for hover mode (initialize before super().__init__)
        self.hover_ctrl = DSLPIDControl(DroneModel.CF2X, g)
        
        # Transition parameters
        self.transition_altitude = 3.5  # m - altitude to transition to cruise
        self.cruise_speed = 10.0  # m/s - target cruise velocity
        self.stable_time = 0.0  # Time spent stable
        self.required_stable_time = 2.0  # seconds - stability required before transition
        
        # Cruise control gains
        self.P_ALTITUDE_CRUISE = 0.4
        self.P_HEADING_CRUISE = 0.6
        self.MAX_BANK_ANGLE = 0.4  # radians (~23 degrees)
        
        # Call parent init (which calls reset)
        super().__init__(drone_model=drone_model, g=g)

    ################################################################################

    def reset(self):
        """Resets the control class."""
        super().reset()
        self.hover_ctrl.reset()
        self.mode = self.HOVER
        self.stable_time = 0.0
        self.last_altitude = 0.0

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the control action (as RPMs) for VTOL drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired angular rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to the 4 motors.
        float
            The current target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the current target torques around each axis.
        """
        self.control_counter += 1
        
        current_altitude = cur_pos[2]
        velocity_magnitude = np.linalg.norm(cur_vel)
        
        # === Mode Transition Logic ===
        if self.mode == self.HOVER:
            # Check conditions for transitioning to cruise
            if current_altitude > self.transition_altitude:
                # Check if velocity is stable (low enough)
                if velocity_magnitude < 0.5:
                    self.stable_time += control_timestep
                else:
                    self.stable_time = 0.0
                
                # Transition after being stable for required time
                if self.stable_time >= self.required_stable_time:
                    self.mode = self.CRUISE
                    print(f"[VTOL] Transitioning to CRUISE mode at altitude {current_altitude:.2f}m")
        
        elif self.mode == self.CRUISE:
            # Revert to hover if altitude drops too low or emergency
            if current_altitude < 2.0:
                self.mode = self.HOVER
                self.stable_time = 0.0
                print(f"[VTOL] Reverting to HOVER mode at altitude {current_altitude:.2f}m")
        
        # === Mode-Based Control ===
        if self.mode == self.HOVER:
            # Use quadcopter hover control for vertical flight
            return self.hover_ctrl.computeControl(
                control_timestep=control_timestep,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=target_rpy,
                target_vel=target_vel,
                target_rpy_rates=target_rpy_rates
            )
        
        else:  # CRUISE mode
            # Calculate desired heading to target
            pos_error = target_pos - cur_pos
            horizontal_distance = np.linalg.norm(pos_error[:2])
            
            # Compute forward velocity vector
            target_vel_cruise = np.zeros(3, dtype=float)
            
            if horizontal_distance > 1.0:
                # Navigate toward target
                direction_2d = pos_error[:2] / max(horizontal_distance, 0.01)
                target_vel_cruise[:2] = direction_2d * self.cruise_speed
            else:
                # Circle around target when close
                cur_rpy = p.getEulerFromQuaternion(cur_quat)
                cur_yaw = cur_rpy[2]
                # Perpendicular velocity for circular motion
                target_vel_cruise[0] = self.cruise_speed * np.cos(cur_yaw + np.pi/2)
                target_vel_cruise[1] = self.cruise_speed * np.sin(cur_yaw + np.pi/2)
            
            # Altitude control
            altitude_error = target_pos[2] - current_altitude
            target_vel_cruise[2] = np.clip(altitude_error * self.P_ALTITUDE_CRUISE, -1.0, 1.0)
            
            # Use hover controller but with forward velocity target
            return self.hover_ctrl.computeControl(
                control_timestep=control_timestep,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=np.zeros(3),
                target_vel=target_vel_cruise,
                target_rpy_rates=np.zeros(3)
            )

    ################################################################################
    
    def get_mode(self):
        """Get current flight mode.
        
        Returns
        -------
        str
            Current mode: 'hover' or 'cruise'
        """
        return self.mode
