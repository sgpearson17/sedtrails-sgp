# Particle seeding in SedTRAILS

Seeding is the process of adding the particle(s) to the model. In SedTRAILS, the type and initial position(s) of the sediment(s) are set as separate populations. The options for particle types are sand, mud, and passive tracer. Multiple strategies are available to locate the paticles as initial conditions. They can be added as either a single particle (point strategy) or more. In case of adding many particles, they can be located along a spcified line (transect strategy), or within a polygon (grid and random strategies). 

Inputs required for particle types:
| Sand       | Mud        | Passive    |
|------------|------------|------------|
| grain size | grain size | grain size |
| ???        | ???        | ???        |


Inputs required for seeding strategies:
| Point         | Transect         | Grid (uniform)    | Grid (random)  |
|---------------|------------------|-------------------|----------------|
| x, y          | x1, y1           | x1, y1            | ???            |
| ???           | x2, y2           | x2, y2            | ???            | 
| ???           | ???              | particle distance | ???            | 
| ???           | ???              | ???               | ???            | 
