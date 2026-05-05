function [BATHY,S] = preprocess_bathymetry(S,XYT)
% Pre-process hydrodynamic model bathymetry

% load bathymetry

% NOTE: ALWAYS USE 'FACE' QUANTITIES (NOT NODE OR EDGE) FROM FM AS PER GUIDANCE FROM JOHAN REYNS

% pre-process data? set to 0 if you want to use pre-computed files
if S.preprocess

    if isempty(S.bathy_filepath)

        switch S.inputModelType
            % =====================================================================
            case 'd3d4' % 'd3d4' = Delft3D-4

                % load main/coarse domain bathymetry data
                fprintf(['%s\n  Loading bed level in water level points']);
                fidMain = qpfopen([S.d3d_path_mainTrimFile]);
                data = qpread(fidMain,'bed level in water level points','griddata',S.subset_t_bed,S.subset_m,S.subset_n);
                data.X = reshape(data.X,[],1);
                data.Y = reshape(data.Y,[],1);
                data.Val = reshape(data.Val,[],1);

                if S.d3d_nested % remove main/coarse domain points where they overlap with nested finer domain
                    inPoly = inpolygon(data.X,data.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
                    data.X(inPoly) = []; % remove coarse domain points inside polygon
                    data.Y(inPoly) = [];
                    data.Val(inPoly,:) = [];

                    % load fine grid cell surface area data
                    fidFine = qpfopen([S.d3d_path_nestedFineTrimFile]);
                    fineData = qpread(fidFine,'bed level in water level points','griddata',S.subset_t_bed,S.subset_m,S.subset_n);
                    fineData.X = reshape(fineData.X,[],1);
                    fineData.Y = reshape(fineData.Y,[],1);
                    fineData.Val = reshape(fineData.Val,[],1);
                    inPoly = inpolygon(fineData.X,fineData.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
                    fineData.X(~inPoly) = []; % remove fine domain points outside polygon
                    fineData.Y(~inPoly) = [];
                    fineData.Val(~inPoly,:) = [];

                    % concatenate fine grid model points to main/coarse grid model points
                    data.X = [data.X; fineData.X];
                    data.Y = [data.Y; fineData.Y];
                    data.Val = [data.Val; fineData.Val];
                    nanPoints = isnan(data.X); % remove NaN points
                    data.X(nanPoints) = [];
                    data.Y(nanPoints) = [];
                    data.Val(nanPoints,:) = [];

                end

                nanPoints = isnan(data.X); % remove NaN points
                data.X(nanPoints) = [];
                data.Y(nanPoints) = [];
                data.Val(nanPoints,:) = [];
                clear nanPoints
                clear fineData

                % =====================================================================
            case 'dfm' % 'dfm' = D-Flow FM
                nn=1;
                % initialize arrays
                output_mesh2d_flowelem_bl = [];

                % load bed level
                data.X = XYT.x;
                data.Y = XYT.y;
                data.Name = 'bedlevel';
                data.Units = 'm';

                switch S.fmOutputType
                    case 'sedtrails_nc'% sedtrails map file
                        fprintf(['%s\n  Loading bedlevel']);
                        bedlevel = squeeze(ncread(S.ncfile,'bedlevel',[1 1],[Inf Inf])); % bedlevel at sedtrails grid corners
                        data.Val = bedlevel(XYT.idx,1); % FIRST TIMESTEP UNTIL JOHAN EXPORTS STATIC

                    case 'merged_nc' % merged netcdf file
                        fprintf(['%s\n  Loading mesh2d_flowelem_bl']);
                        mesh2d_flowelem_bl = squeeze(ncread(S.ncfile,...
                            'mesh2d_flowelem_bl',[1],[Inf])); % "flow element center bedlevel (bl)"
                        data.Val = mesh2d_flowelem_bl(XYT.idx,1);

                    otherwise % standard map file
                        % MAY NEED TO UPDATE THIS!!!
                        fprintf(['%s\n  Loading mesh2d_flowelem_bl']);
                        mesh2d_flowelem_bl = squeeze(ncread(S.ncfile,...
                            'mesh2d_flowelem_bl',[1],[Inf])); % "flow element center bedlevel (bl)"
                        output_mesh2d_flowelem_bl = [output_mesh2d_flowelem_bl; mesh2d_flowelem_bl];
                        fprintf(['%s\n  bl: ' num2str(length(output_mesh2d_flowelem_bl))])
                        if nn == length(S.ncFiles)
                            data.Val = output_mesh2d_flowelem_bl(XYT.idx,1);
                            disp(size(data.Val))
                        end

                end

            case 'xbeach'

                % load bed level
                data.X = XYT.x;
                data.Y = XYT.y;
                data.Name = 'bedlevel';
                data.Units = 'm';

                % read averaged bed level, time-varying. If not static, needs to
                % change
                temp = squeeze(ncread(S.ncfile,...
                    'zb_mean',[1 1 1],[Inf Inf 1]));
                temp = temp(:);
                % for non-static beds:
                %             mesh2d_flowelem_bl = zeros([size(temp,1)*size(temp,2) size(temp,3)]);
                %             for it = 1:size(temp,3)
                %                t2=squeeze(temp(:,:,it)); t2=t2(:);
                %                mesh2d_flowelem_bl(:,it) = t2;
                %             end
                data.Val = temp(XYT.idx);
                clear temp

                % =====================================================================
            otherwise

                inputErrorString = 'Input model type not recognized. Must be';
                error_inputFormat(inputErrorString,supportedModelTypes)
        end

        % Calculate bed slope for purposes of bedslope-induced transport calc
        if S.bedSlope_calculate
            fprintf(['%s\nCalculating bed slopes']);
            % interpolate bathymetry to a regular grid
            [regridX, regridY] = meshgrid(nanmin(data.X):S.bedSlope_dx:nanmax(data.X),...
                nanmin(data.Y):S.bedSlope_dx:nanmax(data.Y));
            regridZ = griddata(data.X,data.Y,data.Val,regridX,regridY);

            % fill NaNs at edges of array with values from adjacent columns and rows
            regridZ = fillmissing(regridZ,'nearest','EndValues','nearest'); % columns
            regridZ = fillmissing(regridZ','nearest','EndValues','nearest')'; % rows

            % calculate local bedslope in x and y direction everywhere
            [dzdx,dzdy] = gradient2(regridX,regridY,regridZ,...
                'discretisation','central'); % NB use 'central' discretization to have it aline with grid coords. there is a border of NaNs though.

            % fill NaNs at edges of array with values from adjacent columns and rows
            dzdx = fillmissing(dzdx,'nearest','EndValues','nearest'); % columns
            dzdx = fillmissing(dzdx','nearest','EndValues','nearest')'; % rows
            dzdy = fillmissing(dzdy,'nearest','EndValues','nearest'); % columns
            dzdy = fillmissing(dzdy','nearest','EndValues','nearest')'; % rows

            % checkplot to make sure that slopes are in the right direction
            if S.bedslope_checkPlot
                figure(9292)
                set(gcf,'Color','w')
                subplot(1,2,1) % dz/dx
                hold on; box on; grid on;
                title('(a) dz/dx')
                pcolor(regridX./1000,regridY./1000,dzdx)
                shading flat

                % plot depth contours
                contour(regridX./1000,regridY./1000,regridZ,[S.plot_MLW,S.plot_MLW],'-k','Linewidth',0.5) % add mean low water contour
                contour(regridX./1000,regridY./1000,regridZ,[S.plot_MHW,S.plot_MHW],'-k','Linewidth',1.0) % add mean high water contour
                for dc = 1:length(S.plot_depthContours)
                    %  add slightly deeper contours
                    contour(regridX./1000,regridY./1000,regridZ,[S.plot_depthContours(dc),S.plot_depthContours(dc)],'-','Linewidth',0.5,'Color',[0.5 0.5 0.5])
                end
                clim([-1 1]./S.bedSlope_caxisFactor);
                cb=colorbar;
                ylabel(cb,'dz/dx [-]','Fontweight', 'bold','FontAngle','italic');
                set(gca,'Layer','top','GridColor',[0.5 0.5 0.5],'GridAlpha',0.4);
                % set axis proportions and limits
                axis equal
                if ~isempty(S.plot_xlim)
                    xlim(S.plot_xlim)
                    ylim(S.plot_ylim)
                end
                xlabel('X [km]');
                ylabel('Y [km]');
                set(gca,'Fontweight', 'bold','FontAngle','italic','FontSize',11);

                subplot(1,2,2) % dz/dy
                hold on; box on; grid on;
                title('(b) dz/dy')
                pcolor(regridX./1000,regridY./1000,dzdy)
                shading flat
                % plot depth contours
                contour(regridX./1000,regridY./1000,regridZ,[S.plot_MLW,S.plot_MLW],'-k','Linewidth',0.5) % add mean low water contour
                contour(regridX./1000,regridY./1000,regridZ,[S.plot_MHW,S.plot_MHW],'-k','Linewidth',1.0) % add mean high water contour
                for dc = 1:length(S.plot_depthContours)
                    %  add slightly deeper contours
                    contour(regridX./1000,regridY./1000,regridZ,[S.plot_depthContours(dc),S.plot_depthContours(dc)],'-','Linewidth',0.5,'Color',[0.5 0.5 0.5])
                end
                clim([-1 1]./S.bedSlope_caxisFactor);
                cb=colorbar;
                ylabel(cb,'dz/dy [-]','Fontweight', 'bold','FontAngle','italic');
                set(gca,'Layer','top','GridColor',[0.5 0.5 0.5],'GridAlpha',0.4);
                % set axis proportions and limits
                axis equal
                if ~isempty(S.plot_xlim)
                    xlim(S.plot_xlim)
                    ylim(S.plot_ylim)
                end
                xlabel('X [km]');
                ylabel('Y [km]');
                set(gca,'Fontweight', 'bold','FontAngle','italic','FontSize',11);

                % Export figure to png file
                pngName = ['Bed Slope Check Plot'];
                dimensions = S.plot_printDimensions.*S.plot_printScaling; % [width height]
                printFigs(dimensions,[S.outputDir filesep pngName],0);

            end

            % interpolate value of dz/dx and dz/dy at all model calculation points
            F = griddedInterpolant(regridX',regridY',dzdx');
            data.dzdx = F(data.X,data.Y);
            clear F
            F = griddedInterpolant(regridX',regridY',dzdy');
            data.dzdy = F(data.X,data.Y);
            clear F
        end

        % OUTPUT BATHYMETRY
        BATHY.filepath = [S.outputDir filesep 'bedlevel.mat'];
        save(BATHY.filepath,'data','-v7.3');
        S.bathy_filepath = BATHY.filepath;

    else
        BATHY.filepath = S.bathy_filepath;

    end

else

    % if a bathymetry file is specified, use that
    if ~isempty(S.bathy_filepath)
        BATHY.filepath = S.bathy_filepath;
    else
        % Assume default filepath
        BATHY.filepath = [S.outputDir filesep 'bedlevel.mat'];
        S.bathy_filepath = BATHY.filepath;
    end

end

end

