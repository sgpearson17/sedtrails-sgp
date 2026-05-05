function [S,O]=SedTRAILS(S0)
% MODEL : SedTRAILS
%
% This model computes Lagrangian sediment pathways due to hydrodynamic forcing in coastal and estuarine environments
% See acknowledgements here: ACKNOWLEDGEMENTS.M
%
% INPUT:
%     S       data structure with fields:
%              .<properties>
%
% v001 - SGP/2024-04-10 - Original file structure based on the code
%       for ShorelineS (IHE Delft & Deltares, 2020):
%       Roelvink, D., Huisman, B., Elghandour, A., Ghonim, M., & Reyns, J.
%       (2020). Efficient modeling of complex sandy coastal evolution at monthly
%       to century time scales. Frontiers in Marine Science, 7, 535.

%
%% Copyright notice
%   --------------------------------------------------------------------
%
%   Copyright (C) 2024 TU Delft & Deltares
%
%   @JOHAN: WHAT SORT OF COPYRIGHT SHOULD WE DO? ALSO GNU? IF WE PUBLISH IN
%   GMD WE NEED TO GO FULLY OPEN (I THINK THIS IS THE WAY TO GO ANYWAY,
%   TOWARDS SOMETHING COMMUNITY-BASED)
%
%   See acknowledgements here: ACKNOWLEDGEMENTS.M
%
%   --------------------------------------------------------------------

fprintf('%s\n','%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('%s\n','%                  RUN SedTRAILS                    %');
fprintf('%s\n','%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('%s\n','STEP 0 : INITIALIZE MODEL');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 0 : SET DEFAULT INPUT PARAMETERS                     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% profile on;
tstart    = now;

[S]         = initialize_defaultsettings(S0);
[S]         = initialize_folderstructure(S);
[S]         = initialize_inputmodel(S);
[PLOTFORMAT]  = initialize_plotFormat(S);


if S.debug==2
    save('debug.mat');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1 : PRE-PROCESS HYDRODYNAMIC MODEL RESULTS           %%
%% create vector fields                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('STEP 1 : PRE-PROCESS HYDRODYNAMIC MODEL RESULTS   \n');

% % in the case of multiple files, loop through each netcdf file and extract output
% for nn = 1:length(ncFiles)
%     S.ncfile = [inputDir filesep S.ncFiles(nn).name];

% TO DO: FIX THIS SO THAT WE CAN COMBINE PARTITIONED NC FILES. RIGHT NOW IT
% IS ONLY FOR SINGLE FILES. JRE: we can do mapmerge on partitioned output,
% so not needed
tic
%% Pre-process hydrodynamic model time and spatial xy-coordinates
[XYT] = preprocess_xyt(S);

%% Pre-process hydrodynamic model bathymetry
[BATHY,S] = preprocess_bathymetry(S,XYT);

%% Pre-process flow velocity vectors
[FLOW] = preprocess_flow(S,XYT);

%% Pre-process sand velocity vectors
[SAND,S] = preprocess_sand(S,XYT,BATHY,FLOW);

%% Pre-process mud velocity vectors
[MUD] = preprocess_mud(S,XYT);

%% Pre-process biological particle vectors (e.g., larvae, mangrove propagules)
[BIO] = preprocess_bio(S,XYT);

%% Pre-process fields for luminescence analysis
[LUM] = preprocess_luminescence(S,XYT);

%% Specify sedtrails start time
[S] = set_sedtrails_starttime(S,XYT);

%% Write metadata prior to running model
save_metadata(S,XYT,BATHY,FLOW,SAND,MUD,BIO,LUM)

elapsedTime = toc;
fprintf(['\nPre-processing duration: ' num2str(elapsedTime./60) ' mins\n']);

if S.debug==2
    save('debug2.mat');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 2 : DEFINE SOURCES                                   %%
%% formerly step 02                                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nSTEP 2 : DEFINING SOURCES    \n');

%% define particle sources
if S.defineSources
    [SRC,S] =  define_sources(S,XYT);
end

%% define Lagrangian coherent structure parameters
% [SRC,LCS] =  define_LCS(S,SRC);

%% define connectivity analysis parameters
if S.connectivity_analyze % analyze connectivity? 1=y 0=n
    [SRC,S,CONNECTIVITY] =  define_connectivity(S,SRC);
else
    CONNECTIVITY = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 3 : COMPUTE & STORE PARTICLE PATHWAYS                %%
%% formerly step 03                                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if S.computePathways
    fprintf('STEP 3 : COMPUTING & STORING PARTICLE PATHWAYS \n');
    % nb. break this out into multiple functions eventually and destroy the
    % mess that is sedtrails_particletracking_cyclic once and for all!

    %% compute & store particle pathways
    [PATHS] = compute_pathways(S, XYT, SRC, BATHY, FLOW, SAND, MUD, BIO);

    %% compute & store Lagrangian Coherent Structures
    [PATHS, LCS] = compute_LCS(S, SRC, BATHY, FLOW, SAND, MUD);

else
    fprintf('SKIPPING STEP 3 AND LOADING EXISTING PARTICLE PATHWAYS \n');

    % FIX THIS!
    PATHS = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 4 : PLOTTING PATHWAYS                                %%
%% formerly step 04                                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot input bathymetry and vector field
[PLOTFORMAT] = plot_input(S, BATHY, FLOW, SAND, MUD, BIO, LUM, PLOTFORMAT);

%% Plot particle pathways
if S.plotPathways
    fprintf('STEP 4 : PLOTTING PATHWAYS \n');
    [PLOTFORMAT] = plot_pathways(S, BATHY, FLOW, SAND, MUD, BIO, LUM, PATHS, CONNECTIVITY, PLOTFORMAT);
else
    fprintf('SKIPPING STEP 4 AND NOT PLOTTING PARTICLE PATHWAYS \n');
end

%% Plot drifter validation data
if S.plotDrifterValidation
    [PLOTFORMAT] = plot_drifterValidation(S, BATHY, FLOW, SAND, MUD, BIO, LUM, PATHS, CONNECTIVITY, PLOTFORMAT);
end

%% Analyze particle pathways
if S.analyzePathways
    fprintf('STEP 4b : ANALYZING PATHWAYS \n');
    analyze_pathways(S, BATHY, FLOW, SAND, MUD, BIO, LUM, PATHS, CONNECTIVITY, PLOTFORMAT);
else
    fprintf('SKIPPING STEP 4b AND NOT ANALYZING PARTICLE PATHWAYS \n');
end

%% Query particle pathways and plot
if S.queryPathways
    if ~isempty(S.query_polygon)
        fprintf(['Querying pathways to/from: ' S.query_name ' \n']);
        [PLOTFORMAT] = query_pathways(S, BATHY, FLOW, SAND, MUD, BIO, LUM, PATHS, CONNECTIVITY, PLOTFORMAT);
    else
        fprintf('S.query_polygon not specified, skipping query analysis...\n')
    end
end
%% Plot light exposure and luminescence
% [PLOTFORMAT, LUM] = plot_luminescence(S, BATHY, FLOW, SAND, MUD, LUM, PATHS, PLOTFORMAT);

%% Plot Lagrangian coherent structures
% [PLOTFORMAT] = plot_LCS(S, BATHY, FLOW, SAND, PATHS, LCS, PLOTFORMAT);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 5 : ANALYZE CONNECTIVITY                             %%
%% compile adjacency matrix to create network                 %%
%% compute connectivity metrics                               %%
%% create plots                                               %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if S.connectivity_analyze % analyze connectivity? 1=y 0=n
    fprintf('STEP 5 : ANALYZING CONNECTIVITY \n');

    %% Compile adjacency matrix to create network
    [S] = compile_adjmat(S, BATHY, FLOW, SAND, MUD, PATHS, PLOTFORMAT);

    %% Compute connectivity metrics
    [CONNECTIVITY] = analyze_connectivity(S, CONNECTIVITY, BATHY, PLOTFORMAT);

    %% Plot networks and connectivity metrics
    [PLOTFORMAT] = plot_connectivity(S, BATHY, FLOW, SAND, MUD, BIO, CONNECTIVITY, PLOTFORMAT);

else
    fprintf('SKIPPING STEP 5 AND NOT ANALYZING CONNECTIVITY \n');
end
%% Store all network data
% save(fullfile(pwd,S.outputdir,'connectivity_output.mat'),'ADJMAT','CONNECTIVITY','S');

%% Conclude run

% profile off;
% profsave(profile('info'),'profile_results')

finalize_metadata(S);

fprintf(['Run completed successfully in ' num2str((now-tstart)*24*60) ' minutes. \n']);

end