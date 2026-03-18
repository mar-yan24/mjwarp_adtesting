"""Inline MJCF XML model definitions for AD testing.

Models are organized by complexity and feature coverage:
- Simple: 1-2 DOF toy models (from grad_test.py)
- Medium: 3-5 DOF chains with mixed joints
- Contact: models with collision geometry enabled
- Free body: 6-DOF floating base models
"""

# ---------------------------------------------------------------------------
# Simple models (from grad_test.py)
# ---------------------------------------------------------------------------

SIMPLE_HINGE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.5 -0.3" qvel="0.1 -0.2" ctrl="0.5 -0.5"/>
  </keyframe>
</mujoco>
"""

SIMPLE_SLIDE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.2" qvel="0.1" ctrl="0.5"/>
  </keyframe>
</mujoco>
"""

THREE_LINK_HINGE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="2"/>
        <body pos="0.3 0 -0.3">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.5 -0.3 0.2" qvel="2.0 -1.0 3.0" ctrl="0.5 -0.5 0.3"/>
  </keyframe>
</mujoco>
"""

SIMPLE_FREE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 1">
      <joint name="j0" type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <keyframe>
    <key qpos="0 0 1 1 0 0 0" qvel="0.1 0 0 0 0.1 0"/>
  </keyframe>
</mujoco>
"""

# ---------------------------------------------------------------------------
# Medium complexity models (new)
# ---------------------------------------------------------------------------

FIVE_LINK_CHAIN_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="1"/>
      <body pos="0 0 -0.3">
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="1"/>
        <body pos="0 0 -0.3">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="1"/>
          <body pos="0 0 -0.3">
            <joint name="j3" type="hinge" axis="0 1 0"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="1"/>
            <body pos="0 0 -0.3">
              <joint name="j4" type="hinge" axis="1 0 0"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
    <motor joint="j3" gear="1"/>
    <motor joint="j4" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.3 -0.2 0.4 -0.1 0.5"
         qvel="1.0 -0.5 0.3 -0.8 0.2"
         ctrl="0.3 -0.4 0.2 -0.1 0.5"/>
  </keyframe>
</mujoco>
"""

SPRING_DAMPER_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0"
             stiffness="10" damping="0.5" springref="0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="1 0 0"
               stiffness="5" damping="0.3" springref="0"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.5 -0.3" qvel="0.1 -0.2" ctrl="0.5 -0.5"/>
  </keyframe>
</mujoco>
"""

FREE_BODY_WITH_ACTUATOR_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 1">
      <joint name="root" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <site name="thruster_x" pos="0.1 0 0"/>
      <site name="thruster_y" pos="0 0.1 0"/>
      <site name="thruster_z" pos="0 0 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <general site="thruster_x" gear="1 0 0 0 0 0" ctrlrange="-1 1"/>
    <general site="thruster_y" gear="0 1 0 0 0 0" ctrlrange="-1 1"/>
    <general site="thruster_z" gear="0 0 1 0 0 0" ctrlrange="-1 1"/>
  </actuator>
  <keyframe>
    <key qpos="0 0 1 1 0 0 0" qvel="0.1 0 0 0 0.1 0" ctrl="0.2 -0.1 0.5"/>
  </keyframe>
</mujoco>
"""

# ---------------------------------------------------------------------------
# Contact models
# ---------------------------------------------------------------------------

CONTACT_SLIDE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.05">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""

CONTACT_SLIDE_DENSE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="dense" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.05">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""

MULTI_CONTACT_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body name="ball1" pos="-0.3 0 0.15">
      <joint name="s1" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body name="ball2" pos="0 0 0.15">
      <joint name="s2" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body name="ball3" pos="0.3 0 0.15">
      <joint name="s3" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="s1" gear="1"/>
    <motor joint="s2" gear="1"/>
    <motor joint="s3" gear="1"/>
  </actuator>
</mujoco>
"""

NO_CONTACT_HIGH_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 2.0">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""

# ---------------------------------------------------------------------------
# Model sets for parametrization
# ---------------------------------------------------------------------------

SMOOTH_MODELS = {
  "simple_hinge": SIMPLE_HINGE_XML,
  "simple_slide": SIMPLE_SLIDE_XML,
  "three_link_hinge": THREE_LINK_HINGE_XML,
  "simple_free": SIMPLE_FREE_XML,
  "five_link_chain": FIVE_LINK_CHAIN_XML,
  "spring_damper": SPRING_DAMPER_XML,
  "free_body_actuator": FREE_BODY_WITH_ACTUATOR_XML,
}

CONTACT_MODELS = {
  "contact_slide": CONTACT_SLIDE_XML,
  "contact_slide_dense": CONTACT_SLIDE_DENSE_XML,
  "multi_contact": MULTI_CONTACT_XML,
  "no_contact_high": NO_CONTACT_HIGH_XML,
}

ALL_MODELS = {**SMOOTH_MODELS, **CONTACT_MODELS}
