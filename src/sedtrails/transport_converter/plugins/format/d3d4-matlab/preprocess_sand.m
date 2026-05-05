function [SAND,S] = preprocess_sand(S,XYT,BATHY,FLOW)
% Pre-process sand velocity vectors

if S.SAND==1
    SAND.used = 1;

    % pre-process data? set to 0 if you want to use pre-computed files
    if S.preprocess
        fprintf('\nPre-processing sand velocities...\n')

        %% READ IN INPUT MODEL DATA
        supportedModelTypes = {'dfm','d3d4','xbeach'};
        switch S.inputModelType
            % =================================================================
            case 'd3d4' % 'd3d4' = Delft3D-4

                % load flow velocity data from Delft3D-4
                [data] = load_flowvelocity_d3d(S,XYT);
                HD.Uc_x = data.XComp;
                clear data.XComp
                HD.Uc_y = data.YComp;
                clear data.YComp

                [data] = load_bedshearstress_d3d(S,XYT);
                HD.mean_bss_mag = data.mean_bss_mag;
                clear data.mean_bss_mag
                HD.max_bss_mag = data.max_bss_mag;
                clear data.max_bss_mag

                % =============================================================
            case 'dfm' % 'dfm' = D-Flow FM

                % load hydrodynamic data from D-Flow FM
                [HD] = load_hydrodynamics_dfm(S,XYT);

                % =============================================================
            case 'xbeach' 

                [HD] = load_hydrodynamics_xbeach(S,XYT,FLOW);

            otherwise
                inputErrorString = 'Input model type not recognized. Must be';
                error_inputFormat(inputErrorString,supportedModelTypes)
        end

        %% CONVERT VECTOR FIELD INTO SAND VELOCITY

        % calculate sand velocity based on chosen formulation
        switch S.sandVelFormulation
            case 'soulsby2011' % Soulsby et al (2011) transport velocities

                % Check whether background grain size is specified; if not assume same as tracer
                if isempty(S.dBackground)
                    S.dBackground = S.dTracer;
                end

                % pre-process data using soulsby2011
                [data] = preprocess_sand_soulsby2011(S,XYT,BATHY,HD);

                % if using bed interaction, override default settings and set initial particle freedom to Soulsby Freedom Factor
                if S.soulsby2011_bedInteraction == 1
                    S.initialParticleFreedom = S.soulsby2011_F0; % initial state of Freedom Factor (0 = trapped, 1 = free) [-]
                end

            case 'pannozzo' % add new formulations here

                % Check whether background grain size is specified; if not assume same as tracer
                if isempty(S.dBackground)
                    S.dBackground = S.dTracer;
                end

                % pre-process data using soulsby2011
                [data] = preprocess_sand_pannozzo(S,XYT,HD);

                % if using bed interaction, override default settings and set initial particle freedom to Soulsby Freedom Factor
                if S.soulsby2011_bedInteraction == 1
                    S.initialParticleFreedom = S.soulsby2011_F0; % initial state of Freedom Factor (0 = trapped, 1 = free) [-]
                end

            case 'vanwesten' % add new formulations here

                % Check whether background grain size is specified; if not assume same as tracer
                if isempty(S.dBackground)
                    S.dBackground = S.dTracer;
                end

                % pre-process data using soulsby2011
                [data] = preprocess_sand_vanwesten(S,XYT,HD);

                % if using bed interaction, override default settings and set initial particle freedom to Soulsby Freedom Factor
                if S.soulsby2011_bedInteraction == 1
                    S.initialParticleFreedom = S.soulsby2011_F0; % initial state of Freedom Factor (0 = trapped, 1 = free) [-]
                end

            otherwise
                inputErrorString = 'Sand velocity formulation not recognized. Must be';
                error_inputFormat(inputErrorString,S.supportedSandTransportFormulations)

        end

        % OUTPUT SAND DATA
        fprintf('Saving sand velocities...\n')
        SAND.filepath = [S.outputDir filesep 'sand_velocity.mat'];
        [mg,ng]=size(data.X);
        SAND.boundaries = [1 mg 1 ng]; % see if you can delete this; legacy code
        save(SAND.filepath,'data','-v7.3');
    else

        % Assume default filepath
        SAND.boundaries = [1 1 1 1]; % see if you can delete this; legacy code
        SAND.filepath = [S.outputDir filesep 'sand_velocity.mat'];
    end

else
    % OUTPUT BLANK STRUCTURE
    SAND.used = 0;
    SAND.filepath = [];

end

end
