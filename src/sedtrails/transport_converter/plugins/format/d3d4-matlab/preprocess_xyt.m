function [XYT] = preprocess_xyt(S)
% Pre-process hydrodynamic model time and spatial xy-coordinates

% NOTE: ALWAYS USE 'FACE' QUANTITIES (NOT NODE OR EDGE) AS PER GUIDANCE FROM JOHAN REYNS

% pre-process data? set to 0 if you want to use pre-computed files
if S.preprocess

    supportedModelTypes = {'dfm','d3d4','xbeach'};
    switch S.inputModelType
        % =====================================================================
        case 'd3d4' % 'd3d4' = Delft3D-4

            fidMain = qpfopen([S.d3d_path_mainTrimFile]);

            % load coarse depth-averaged velocity data
            fprintf(['%s\n  Loading depth averaged velocity to read time']);
            data = qpread(fidMain,'depth averaged velocity','griddata',S.subset_t,S.subset_m,S.subset_n);
            data.X = reshape(data.X,[],1);
            data.Y = reshape(data.Y,[],1);
            data.Time;

            if S.d3d_nested
                % remove main/coarse domain points where they overlap with nested finer domain
                inPoly = inpolygon(data.X,data.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
                data.X(inPoly) = [];
                data.Y(inPoly) = [];

                % load fine depth-averaged velocity data
                fidFine = qpfopen([S.d3d_path_nestedFineTrimFile]);
                fprintf(['%s\n  Loading depth averaged velocity']);
                fineData = qpread(fidFine,'depth averaged velocity','griddata',S.subset_t,S.subset_m,S.subset_n);
                fineData.X = reshape(fineData.X,[],1);
                fineData.Y = reshape(fineData.Y,[],1);

                % remove fine domain points outside polygon
                inPoly = inpolygon(fineData.X,fineData.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
                fineData.X(~inPoly) = [];
                fineData.Y(~inPoly) = [];

                % concatenate fine model points to coarse model points
                data.X = [data.X; fineData.X];
                data.Y = [data.Y; fineData.Y];
                clear nanPoints
                clear fineData

            end

            % remove NaN points
            nanPoints = isnan(data.X);
            data.X(nanPoints) = [];
            data.Y(nanPoints) = [];

            % =====================================================================
        case 'dfm' % 'dfm' = D-Flow FM

            % initialize output arrays
            input_time = [];
            input_net_xcc = [];
            input_net_ycc = [];

            % load time data
            fprintf('%s\n  Loading time data');
            timeOffset = ncreadatt(S.ncfile,'time','units'); % extract start time from units attribute of time series
            timeOffset = posixtime(datetime(timeOffset(15:34))); % netcdf stores everything in seconds relative to posixtime (1970-01-01 00:00:00)
            input_time =  datenum(datetime(squeeze(ncread(S.ncfile,'time',[1],[Inf]))+timeOffset, 'ConvertFrom', 'posixtime')); % time
            data.Time = input_time(S.subset_t);

            % TO DO: FIX THIS SO THAT WE CAN COMBINE PARTITIONED NC FILES. RIGHT NOW IT
            % IS ONLY FOR SINGLE FILES
            nn=1;

            %----------------------------------------------------------------------
            % X/Y/Z COORDINATES
            %----------------------------------------------------------------------

            % load xy coordinates
            if strcmp(S.ncfile(end-5:end),'map.nc') % standard map file
                fprintf(['%s\n  Loading mesh2d_face_x/y']);
                mesh2d_face_x = squeeze(ncread(S.ncfile,...
                    'mesh2d_face_x',[1],[Inf])); % "x-coordinate of cell faces"
                mesh2d_face_y = squeeze(ncread(S.ncfile,...
                    'mesh2d_face_y',[1],[Inf])); % "y-coordinate of cell faces"
                input_net_xcc = [input_net_xcc; mesh2d_face_x];
                input_net_ycc = [input_net_ycc; mesh2d_face_y];
                clear mesh2d_face_x mesh2d_face_y
            else % sedtrails map file
                fprintf(['%s\n  Loading net_xcc/ycc']);
                net_xcc = squeeze(ncread(S.ncfile,...
                    'net_xcc',[1],[Inf])); % "x-coordinate of sedtrails grid corner"
                net_ycc = squeeze(ncread(S.ncfile,...
                    'net_ycc',[1],[Inf])); % "y-coordinate of sedtrails grid corner"
                input_net_xcc = [input_net_xcc; net_xcc];
                input_net_ycc = [input_net_ycc; net_ycc];
                clear net_xcc net_ycc
            end
            totalSize = length(input_net_xcc);

            % determine sedtrails spatial domain to extract from map files
            if nn == length(S.ncFiles)
                if S.useFullModelExtents
                    XYT.idx = true(size(input_net_xcc)); % if there's no bounding polygon, use the whole thing
                else
                    % read in bounding polygon
                    if ~isempty(S.sedtrailsDomainPol)
                        sedtrailsDomain=landboundary('read', S.sedtrailsDomainPol);
                        XYT.idx= inpolygon(input_net_xcc, input_net_ycc, sedtrailsDomain(:,1), sedtrailsDomain(:,2));
                    else % otherwise form a bounding polygon from x/y extents
                        sedtrailsDomain = [S.subset_x(1) S.subset_y(1); ...
                            S.subset_x(2) S.subset_y(1); ...
                            S.subset_x(2) S.subset_y(2); ...
                            S.subset_x(1) S.subset_y(2); ...
                            S.subset_x(1) S.subset_y(1)];
                        XYT.idx= inpolygon(input_net_xcc, input_net_ycc, sedtrailsDomain(:,1), sedtrailsDomain(:,2));
                    end
                end

                % select points within chosen sedtrails domain
                input_net_xcc = input_net_xcc(XYT.idx);
                input_net_ycc = input_net_ycc(XYT.idx);
            end

            disp(['Using ' num2str(length(S.subset_t)) ' of ' num2str(length(input_time)) ' timesteps and ' ...
                num2str(length(input_net_xcc)) ' of ' num2str(totalSize) ' grid points'])

            % load bed level
            data.X = input_net_xcc;
            data.Y = input_net_ycc;

        case 'xbeach'
            % initialize output arrays
            input_net_xcc = [];
            input_net_ycc = [];

            % load time data
            t0 = datenum(2020,1,1,0,0,0);     % xbeach has no reference date
            t  = ncread(S.ncfile,'meantime');
            input_time = t0 + t/24/3600;
            data.Time = input_time(S.subset_t);

            %----------------------------------------------------------------------
            % X/Y/Z COORDINATES
            %----------------------------------------------------------------------
            nn=1;
            % load xy coordinates
            net_xcc = squeeze(ncread(S.ncfile,...
                'globalx',[1 1],[Inf Inf]));
            net_ycc = squeeze(ncread(S.ncfile,...
                'globaly',[1 1],[Inf Inf]));
            input_net_xcc = [input_net_xcc; net_xcc(:)];
            input_net_ycc = [input_net_ycc; net_ycc(:)];
            clear net_xcc net_ycc

            % determine sedtrails spatial domain to extract from map files
            if nn == length(S.ncFiles)
                if S.useFullModelExtents
                    XYT.idx = true(size(input_net_xcc)); % if there's no bounding polygon, use the whole thing
                else
                    % read in bounding polygon
                    if ~isempty(S.sedtrailsDomainPol)
                        sedtrailsDomain=landboundary('read', S.sedtrailsDomainPol);
                        XYT.idx= inpolygon(input_net_xcc, input_net_ycc, sedtrailsDomain(:,1), sedtrailsDomain(:,2));
                    else % otherwise form a bounding polygon from x/y extents
                        sedtrailsDomain = [S.subset_x(1) S.subset_y(1); ...
                            S.subset_x(2) S.subset_y(1); ...
                            S.subset_x(2) S.subset_y(2); ...
                            S.subset_x(1) S.subset_y(2); ...
                            S.subset_x(1) S.subset_y(1)];
                        XYT.idx= inpolygon(input_net_xcc, input_net_ycc, sedtrailsDomain(:,1), sedtrailsDomain(:,2));
                    end
                end

                % select points within chosen sedtrails domain
                input_net_xcc = input_net_xcc(XYT.idx);
                input_net_ycc = input_net_ycc(XYT.idx);
            end

            % load bed level
            data.X = input_net_xcc;
            data.Y = input_net_ycc;
            % =====================================================================
        otherwise
            inputErrorString = 'Error: Input model type not recognized. Must be';
            error_inputFormat(inputErrorString,supportedModelTypes)
    end

    % =====================================================================

    % SAVE TIME
    XYT.t = data.Time;

    % SAVE XY-COORDINATES
    XYT.x = data.X;
    XYT.y = data.Y;

    % OUTPUT FLOW DATA
    XYT.filepath = [S.outputDir filesep 'xyt.mat'];
    fprintf(['Saving ' strrep(XYT.filepath,'\','/') '\n']);
    save(XYT.filepath,'data','-v7.3');
    clear data;

    clear data;


else
    load([S.outputDir filesep 'xyt.mat']);
    % SAVE TIME
    XYT.t = data.Time;

    % SAVE XY-COORDINATES
    XYT.x = data.X;
    XYT.y = data.Y;

end

end