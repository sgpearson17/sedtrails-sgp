function [XY] = preprocess_xy(S)
% Pre-process hydrodynamic model spatial xy-coordinates

% load bathymetry

% NOTE: ALWAYS USE 'FACE' QUANTITIES (NOT NODE OR EDGE) AS PER GUIDANCE FROM JOHAN REYNS

switch S.inputModelType
    % =====================================================================
    case 'd3d4' % 'd3d4' = Delft3D-4
        error('Sorry, Delft3D-4 support not yet added');
    
    % =====================================================================
    case 'dfm' % 'dfm' = D-Flow FM

        % initialize output arrays
        output_net_xcc = [];
        output_net_ycc = [];

        % TO DO: FIX THIS SO THAT WE CAN COMBINE PARTITIONED NC FILES. RIGHT NOW IT
        % IS ONLY FOR SINGLE FILES
        nn=1;

        %----------------------------------------------------------------------
        % X/Y/Z COORDINATES
        %----------------------------------------------------------------------

        % load xy coordinates
        if strcmp(S.ncfile(end-5:end),'map.nc') % standard map file
            mesh2d_face_x = squeeze(ncread(S.ncfile,...
                'mesh2d_face_x',[1],[Inf])); % "x-coordinate of cell faces"
            mesh2d_face_y = squeeze(ncread(S.ncfile,...
                'mesh2d_face_y',[1],[Inf])); % "y-coordinate of cell faces"
            output_net_xcc = [output_net_xcc; mesh2d_face_x];
            output_net_ycc = [output_net_ycc; mesh2d_face_y];
            disp(['x: ' num2str(length(output_net_xcc))])
            clear mesh2d_face_x mesh2d_face_y
        else % sedtrails map file
            net_xcc = squeeze(ncread(S.ncfile,...
                'net_xcc',[1],[Inf])); % "x-coordinate of sedtrails grid corner"
            net_ycc = squeeze(ncread(S.ncfile,...
                'net_ycc',[1],[Inf])); % "y-coordinate of sedtrails grid corner"
            output_net_xcc = [output_net_xcc; net_xcc];
            output_net_ycc = [output_net_ycc; net_ycc];
            clear net_xcc net_ycc
        end

        % determine sedtrails spatial domain to extract from map files
        if nn == length(S.ncFiles)
            if ~isempty(S.sedtrailsDomainPol)
                sedtrailsDomain=landboundary('read', S.sedtrailsDomainPol);
                XY.idx= inpolygon(output_net_xcc, output_net_ycc, sedtrailsDomain(:,1), sedtrailsDomain(:,2));
            else
                XY.idx = true(size(output_net_xcc)); % if there's no polygon, use the whole thing
            end

            % select points within chosen sedtrails domain
            output_net_xcc = output_net_xcc(S.idx);
            output_net_ycc = output_net_ycc(S.idx);
        end

        % load bed level
        data.X = output_net_xcc;
        data.Y = output_net_ycc;


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


% OUTPUT XY-COORDINATES
XY.x = data.X;
XY.y = data.Y;

end