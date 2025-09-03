# Tutorial 1: Running a Basic Simulation


1. From the installation directory, copy/paste the exmaples folder (i.e., .../sedtrails/examples) in a directory where you want to run the model.
```bash
cp -r .../sedtrails/examples <your runs folder>
```

2. Download the dataset file named "inlet_sedtrails.nc" from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm).

3. Go to the directory in which you want to run the model:
```bash
cd .../<your runs folder>
```

4. In the file config.example.yaml, update the directory for "input_data" to read the data-set you doanloaded in the first step: 
  input_data: ./sample-data/inlet_sedtrails.nc

5. Run the model:
```bash
sedtrails run -c ./config.example.yaml
```

6. The results plots should pop up and should be saved in the ./results directory.

::: note
The linux users should replace / with \
:::
