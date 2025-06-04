# Sample Data

Data for testing purposes.

## Input datasets

| Dataset | Description | Source |
|----------|----------|----------|
| d3d4   | Format Binary trim-*.dat  | Delft3D-4   |
| dfm    | NetCDF file   | D-Flow   |
| sedtrails | A dataset to test in developement. It contains inputs and expected outputs for several three use cases: passive, sand, single and multiple particles. Read below for details. | SedTRAILS Matlab | 


<a href="https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9">
<button type="button">
    Download Datasets
</button>
</a>


## SedTRAILS Dataset
The inputs and outputs in this dataset  are based on the simple inlet model in D-Flow FM, using `\SedTRAILS_InputExamples\dfm\inlet_sedtrails.nc`
Three output datasets were creted uisng the following inputs, one per input:

1. ``SedTRAILS_input_inletFM_SinglePassiveParticleExample_v001.m``
2. ``SedTRAILS_input_inletFM_SingleSandParticleExample_v001.m``
3. ``SedTRAILS_input_inletFM_MultipleSandParticleExample_v001.m``

The above cases were run using the matlab code in this folder:
``\SedTRAILS_InputExamples\sedtrails\sedtrails_repo\trunk``
Each run took only a few seconds on my laptop:
1. Passive particle x1: 1.81 s
2. Sand particle x1: 1.79 s
3. Sand particles x242 (500m grid): 6.42 s

The output can be found here:
``\SedTRAILS_InputExamples\sedtrails\output``

### Single Passive Particle:
![Image](https://github.com/user-attachments/assets/60138d34-f30b-4989-a5f5-20c3cc3c1170)

### Single Sand Particle (500 um):
![Image](https://github.com/user-attachments/assets/65e61a55-0243-4f6c-9122-65f42d64f018)

### Multiple Sand Particles (500 um):
![Image](https://github.com/user-attachments/assets/b1865e18-75dd-48e2-99ff-f868b9f8c9a0)
