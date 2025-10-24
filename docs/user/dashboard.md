# Monitoring Dashboard

The purpose of the dashboard is to provide a real-time visualization of the simulation as it runs. It can help users to monitor the progress of the simulation, identify potential issues, and gain insights into sediment transport and deposition. 

:::warning
There is a [known issue](https://github.com/sedtrails/sedtrails/issues/355) where output is not written when the dashboard is used. This will be addressed soon.
:::

![sedtrails development workflow](../_static\img\dashboard-example.png)

## Navigating the Dashboard

1. In the top left panel we see the latest flow field timestep. The colour scale corresponds to the magnitude of the flow field and vectors indicating direction are displayed on top.
2. In the bottom left panel, we see the bathymetry and particle positions overlaid. Initial positions are marked with Xs, and the position at the current timestep is indicated with a circle.
3. The top right panel shows the average and maximum magnitude (?) of the alongshore velocity of all particles. For now it is assumed that the alongshore velocity is in the X direction.
4. The next panel shows the average and maximum magnitude (?) of the cross-shore velocity of all particles. For now it is assumed that the cross-shore velocity is in the Y direction.
5. The next panel average distance per timestep indicates the average distance travelled by all particles in each timestep. This is perhaps the most important diagnostic tool: if the travel distance of your particles is zero and you expect them to move (given your particle characteristics and flow conditions), something may be wrong!
6. The bottom right panel indicates the average and maximum depths at which particles are buried ($\delta_{burial}$) and the mixing depth ($\delta_{mix}$ in [van Westen et al (2025)](https://doi.org/10.1038/s41598-025-92910-z))
7. The progress bar at the bottom indicates the current simulation timestep and the relative percentage of your total run that is complete.

Future implementations will allow for more customization of the dashboard and optimize it for speed. However, it is primarily intended as an "is my model actually running as I intend it?" diagnostic, so users are advised to use the model output files and other visualization utilities to plot their results.

## Turning Off the Dashboard
At the moment, the dashboard has been included as a quick check and has not been optimized for speed. Once you are satisfied that the model is running, you may wish to disable the dashboard for "production runs". To do so, change ``enable: true`` to ``enable: false`` in the config file ([see here](../user/simulations.md)):

```yaml
visualization:
  dashboard:
    enable: true
    update_interval: 1H
```
It is also possible to change the interval at which the dashboard is updated by modifying ``update_interval: 1H``. The default is 1 hour because of the timescale of our example simulation, but it may be useful to change this if your simulation is much longer/shorter.