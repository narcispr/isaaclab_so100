import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Construct the absolute path to the USD file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VALVE_USD_PATH = os.path.join(_THIS_DIR, "asset", "valve2.usd")

# Create Articulation configuration for the valve
VALVE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=VALVE_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"valve_joint": 0.0},    # Start at closed position
        joint_vel={"valve_joint": 0.0},
    ),
    actuators={
        "valve_handle_actuator": ImplicitActuatorCfg(
            joint_names_expr=["valve_joint"],
            effort_limit=2.0,
            velocity_limit_sim=1.5,
            stiffness=10.0,    # Light spring-like return
            damping=2.0,       # Simulate joint friction
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the valve object with a rotatable handle."""
