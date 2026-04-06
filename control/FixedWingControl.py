import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class FixedWingControl(BaseControl):
    """Fixed-wing control class with simplified aerodynamic model.
    
    Implements a basic autopilot for fixed-wing UAVs that:
    - Maintains minimum forward velocity (stall prevention)
    - Uses elevator for pitch/altitude control
    - Uses ailerons for bank-to-turn navigation
    - Maps control inputs to motor RPMs for PyBullet compatibility
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Initialize fixed-wing controller.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (should be DroneModel.FIXEDWING).
        g : float, optional
            The gravitational acceleration in m/s^2.
        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.FIXEDWING:
            print("[ERROR] in FixedWingControl.__init__(), FixedWingControl requires DroneModel.FIXEDWING")
            exit()
        
        # Flight characteristics
        self.MIN_SPEED = 8.0  # m/s - stall speed
        self.CRUISE_SPEED = 12.0  # m/s - optimal cruise
        self.MAX_SPEED = 20.0  # m/s - maximum speed
        
        # Control gains for altitude and heading
        self.P_ALTITUDE = 0.5
        self.D_ALTITUDE = 0.3
        self.P_HEADING = 0.8
        self.MAX_PITCH_ANGLE = 0.35  # radians (~20 degrees)
        self.MAX_BANK_ANGLE = 0.52  # radians (~30 degrees)
        
        # Turn radius at cruise speed
        self.MIN_TURN_RADIUS = 10.0  # meters
        
        # Motor distribution (single pusher prop + control surfaces simulated)
        self.BASE_THROTTLE = 0.7  # Baseline throttle for cruise
        
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control class.

        Integral errors and previous states are zeroed.
        """
        super().reset()
        self.last_altitude_error = 0.0
        self.last_heading_error = 0.0
        self.integral_altitude = 0.0
        self.waypoint_index = 0

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
        """Computes the control action (as RPMs) for fixed-wing drone.

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
        
        # Get current orientation
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        cur_roll, cur_pitch, cur_yaw = cur_rpy
        
        # Compute direction to target
        pos_error = target_pos - cur_pos
        distance_to_target = np.linalg.norm(pos_error[:2])  # XY distance
        
        # Desired heading (yaw) to target
        desired_yaw = math.atan2(pos_error[1], pos_error[0])
        heading_error = self._wrap_angle(desired_yaw - cur_yaw)
        
        # Altitude control (pitch adjustment)
        altitude_error = target_pos[2] - cur_pos[2]
        altitude_error_rate = (altitude_error - self.last_altitude_error) / control_timestep
        self.last_altitude_error = altitude_error
        
        # Desired pitch (for altitude control)
        desired_pitch = np.clip(
            self.P_ALTITUDE * altitude_error + self.D_ALTITUDE * altitude_error_rate,
            -self.MAX_PITCH_ANGLE,
            self.MAX_PITCH_ANGLE
        )
        
        # Bank angle for turning (proportional to heading error)
        desired_roll = np.clip(
            self.P_HEADING * heading_error,
            -self.MAX_BANK_ANGLE,
            self.MAX_BANK_ANGLE
        )
        
        # Compute current forward speed
        forward_vel = np.linalg.norm(cur_vel[:2])
        
        # Throttle control: maintain cruise speed
        speed_error = self.CRUISE_SPEED - forward_vel
        throttle = np.clip(self.BASE_THROTTLE + 0.1 * speed_error, 0.3, 1.0)
        
        # If speed is too low, increase throttle significantly (stall prevention)
        if forward_vel < self.MIN_SPEED:
            throttle = 1.0
            desired_pitch = max(desired_pitch, -0.1)  # Nose down slightly to gain speed
        
        # Convert throttle and control surface deflections to RPM commands
        # For compatibility with the 4-motor interface, we distribute control
        base_rpm = throttle * self.MAX_RPM
        
        # Roll control: differential "thrust" on left/right
        roll_adjustment = desired_roll * 0.2 * self.MAX_RPM
        
        # Pitch control: differential thrust fore/aft
        pitch_adjustment = desired_pitch * 0.15 * self.MAX_RPM
        
        # Motor mapping (treating as X-configuration for control surfaces)
        # Motor 0 (front-right): pitch up, roll right
        # Motor 1 (front-left): pitch up, roll left  
        # Motor 2 (rear-left): pitch down, roll left
        # Motor 3 (rear-right): pitch down, roll right
        rpm_0 = base_rpm + pitch_adjustment + roll_adjustment
        rpm_1 = base_rpm + pitch_adjustment - roll_adjustment
        rpm_2 = base_rpm - pitch_adjustment - roll_adjustment
        rpm_3 = base_rpm - pitch_adjustment + roll_adjustment
        
        rpm = np.array([rpm_0, rpm_1, rpm_2, rpm_3])
        rpm = np.clip(rpm, 0, self.MAX_RPM)
        
        # Compute thrust and torques for return values
        thrust = base_rpm**2 * self.KF * 4  # Total thrust approximation
        torques = np.array([
            roll_adjustment * self.KF,
            pitch_adjustment * self.KF,
            0.0  # Yaw torque (minimal for fixed-wing)
        ])
        
        return rpm, thrust, torques

    ################################################################################

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi] range.
        
        Parameters
        ----------
        angle : float
            Angle in radians.
            
        Returns
        -------
        float
            Wrapped angle in [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    ################################################################################
    
    @property
    def MAX_RPM(self):
        """Maximum RPM for the fixed-wing's motor.
        
        Returns
        -------
        float
            Maximum RPM value.
        """
        # Compute from URDF parameters
        return np.sqrt((self._getURDFParameter('thrust2weight') * self.GRAVITY) / (4 * self.KF))
