# xmls

These are mujoco XML files which are used as bases for the simulations.

Some design goals for them:

- XML should be complete and simulate-able as-is
    - Include a floor geom which is a plane
    - Include joint sensor for the robot which provide observation
    - Include actuators which provide control
- Default positions should all be neutral
    - position 0,0,0 should be resting on the floor, not intersecting it
    - robot should start at the origin
- Scene should be clear of other objects
    - no obstacles or things to manipulate
    - only the robot in the scene

Requirements for the robot
- Position joints should be separate and named `x`, `y`, and `z`
- 0, 0, 0 position should be resting on the floor above the origin at a neutral position
- First 6 sensors should be (in order):
    - joint positions for x, y, z (absolute position in the scene)
    - joint velocities for x, y, z (absolute velocity in the scene)
