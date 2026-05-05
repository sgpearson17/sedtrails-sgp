function [FLOW] = preprocess_flow(S,XYT)
% Pre-process flow velocity vectors

if S.FLOWVEL==1
    FLOW.used = 1;

    % pre-process data? set to 0 if you want to use pre-computed files
    if S.preprocess

        supportedModelTypes = {'dfm','d3d4','xbeach'};
        switch S.inputModelType
            % =================================================================
            case 'd3d4' % 'd3d4' = Delft3D-4

                % Load and reformat flow velocity

                data = load_flowvelocity_d3d(S); % note that you do not use data now (i.e. it is not an output parameter)

                % save additional variables
                data.Time = XYT.t;
                data.X = XYT.x;
                data.Y = XYT.y;

                % =============================================================
            case 'dfm' % 'dfm' = D-Flow FM

                % initialize arrays
                output_sea_water_x_velocity = [];
                output_sea_water_y_velocity = [];

                % TO DO: FIX THIS SO THAT WE CAN COMBINE PARTITIONED NC FILES. RIGHT NOW IT
                % IS ONLY FOR SINGLE FILES
                % JRE: not needed, you can run mapmerge on output
                nn=1;

                % Load and reformat flow velocity
                data.Name = 'flow_velocity';
                data.Time = XYT.t;
                data.X = XYT.x;
                data.Y = XYT.y;

                % use correct flow velocity field name
                fprintf(['Loading ' strrep(S.ncfile,'\','/') '\n']);
                if strcmp(S.ncfile(end-5:end),'ils.nc') % if '*sedtrails.nc'
                    flowVelFieldNameX = 'sea_water_x_velocity';
                    flowVelFieldNameY = 'sea_water_y_velocity';
                else % normal fm map file
                    flowVelFieldNameX = 'mesh2d_ucx';
                    flowVelFieldNameY = 'mesh2d_ucy';
                end
              

                % save flow velocity
                if nn == length(S.ncFiles)
                    % x-velocity
                    fprintf(['%s\n','Loading ' flowVelFieldNameX]);
                    [data.XComp] = read_ncdata_spacetime(S,XYT,flowVelFieldNameX);
%                     output_sea_water_x_velocity = squeeze(ncread(S.ncfile,...
%                     flowVelFieldNameX,[1 1],[Inf Inf])); % depth-averaged velocity on grid corner, x-component
%                     data.XComp = output_sea_water_x_velocity(XYT.idx,S.subset_t);
%                     clear output_sea_water_x_velocity 

                    % y velocity
                    fprintf(['%s\n','Loading ' flowVelFieldNameY]);
                    [data.YComp] = read_ncdata_spacetime(S,XYT,flowVelFieldNameY);
%                     output_sea_water_y_velocity = squeeze(ncread(S.ncfile,...
%                     flowVelFieldNameY,[1 1],[Inf Inf])); % depth-averaged velocity on grid corner, y-component
%                     data.YComp = output_sea_water_y_velocity(XYT.idx,S.subset_t);
%                     clear output_sea_water_y_velocity
                end

                %  clear sea_water_x_velocity sea_water_y_velocity 
                clear flowVelFieldNameX flowVelFieldNameY

            case 'xbeach'

                % initialize arrays
                output_sea_water_x_velocity = [];
                output_sea_water_y_velocity = [];

                % Load and reformat flow velocity
                data.Name = 'flow_velocity';
                data.Time = XYT.t;
                data.X = XYT.x;
                data.Y = XYT.y;

                % use correct flow velocity field name. These are the time-averaged GLM
                % velocities
                fprintf(['Loading ' strrep(S.ncfile,'\','/') '\n']);
                flowVelFieldNameX = 'u_mean';
                flowVelFieldNameY = 'v_mean';

                % x-velocity
                temp = squeeze(ncread(S.ncfile,...
                flowVelFieldNameX,[1 1 1],[Inf Inf Inf]));
                output_sea_water_x_velocity = zeros([size(temp,1)*size(temp,2) size(temp,3)]);
                for it = 1:size(temp,3)
                   t2=squeeze(temp(:,:,it)); t2=t2(:);
                   output_sea_water_x_velocity(:,it) = t2;
                end
                data.XComp = output_sea_water_x_velocity(XYT.idx,S.subset_t);
                clear output_sea_water_x_velocity
                % y velocity
                temp = squeeze(ncread(S.ncfile,...
                flowVelFieldNameY,[1 1 1],[Inf Inf Inf]));
                output_sea_water_x_velocity = zeros([size(temp,1)*size(temp,2) size(temp,3)]);
                for it = 1:size(temp,3)
                   t2=squeeze(temp(:,:,it)); t2=t2(:);
                   output_sea_water_x_velocity(:,it) = t2;
                end
                data.XComp = output_sea_water_x_velocity(XYT.idx,S.subset_t);
                clear output_sea_water_x_velocity

                clear flowVelFieldNameX flowVelFieldNameY                

                % =============================================================
            otherwise
                inputErrorString = 'Input model type not recognized. Must be';
                error_inputFormat(inputErrorString,supportedModelTypes)
        end

        % OUTPUT FLOW DATA
        FLOW.filepath = [S.outputDir filesep 'instantaneous flow velocity.mat'];
        [mg,ng]=size(data.X);
        FLOW.boundaries = [1 mg 1 ng]; % see if you can delete this; legacy code
        fprintf(['Saving ' strrep(FLOW.filepath,'\','/') '\n']);
        save(FLOW.filepath,'data','-v7.3');
        clear data;
    else

        % Assume default filepath
        FLOW.filepath = [S.outputDir filesep 'instantaneous flow velocity.mat'];
        FLOW.boundaries = [1 1 1 1]; % see if you can delete this; legacy code

    end
else
    % OUTPUT BLANK STRUCTURE
    FLOW.used = 0;
    FLOW.filepath = [];
end

end