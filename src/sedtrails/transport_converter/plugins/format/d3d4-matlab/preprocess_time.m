function [TIME] = preprocess_time(S)
% Pre-process hydrodynamic model time

% identify which type of model is being used as input
switch S.inputModelType
    % =====================================================================
    case 'd3d4' % 'd3d4' = Delft3D-4

        error('Sorry! I haven''t updated this part of the code for D3D4 yet!')

    % =====================================================================
    case 'dfm' % 'dfm' = D-Flow FM

        % initialize time array
        timeOffset = ncreadatt(S.ncfile,'time','units'); % extract start time from units attribute of time series
        timeOffset = posixtime(datetime(timeOffset(15:34))); % netcdf stores everything in seconds relative to posixtime (1970-01-01 00:00:00)
        time =  datenum(datetime(squeeze(ncread(S.ncfile,'time',[1],[Inf]))+timeOffset, 'ConvertFrom', 'posixtime')); % time

        % OUTPUT TIME
        TIME.t = time(S.subset_t);

    % =====================================================================
    otherwise
        inputErrorString = 'Error: Input model type not recognized. Must be';
        for ii = 1:length(S.supportedModelTypes)
            if ii==1
                inputErrorString = [inputErrorString ' ''' S.supportedModelTypes{ii} ''''];
            else
                inputErrorString = [inputErrorString ' or ''' S.supportedModelTypes{ii} ''''];
            end
        end
        error(inputErrorString);

end

end