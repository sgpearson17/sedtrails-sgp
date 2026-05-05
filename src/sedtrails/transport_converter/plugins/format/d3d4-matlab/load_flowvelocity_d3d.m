function [data] = load_flowvelocity_d3d(S,XYT)
% Load flow velocity data from d3d-4 model

% open main trim file
fidMain = qpfopen([S.d3d_path_mainTrimFile]);

%% LOAD VELOCITY VECTOR FIELD
% either depth-averaged or from a specific vertical layer
if S.d3d_depAvgFlow % depth-averaged flow

    % load coarse depth-averaged velocity data
    data = qpread(fidMain,'depth averaged velocity','griddata',S.subset_t,S.subset_m,S.subset_n);
    data.X = reshape(data.X,[],1);
    data.Y = reshape(data.Y,[],1);
    data.XComp = reshape(data.XComp,length(data.Time),[])';
    data.YComp = reshape(data.YComp,length(data.Time),[])';

    % add nested points if applicable
    if S.d3d_nested
        % remove main/coarse domain points where they overlap with nested finer domain
        inPoly = inpolygon(data.X,data.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
        data.X(inPoly) = [];
        data.Y(inPoly) = [];
        data.XComp(inPoly,:) = [];
        data.YComp(inPoly,:) = [];

        % load fine depth-averaged velocity data
        fidFine = qpfopen([S.d3d_path_nestedFineTrimFile]);
        fineData = qpread(fidFine,'depth averaged velocity','griddata',S.subset_t,S.subset_m,S.subset_n);
        fineData.X = reshape(fineData.X,[],1);
        fineData.Y = reshape(fineData.Y,[],1);
        fineData.XComp = reshape(fineData.XComp,length(fineData.Time),[])';
        fineData.YComp = reshape(fineData.YComp,length(fineData.Time),[])';

        % remove fine domain points outside polygon
        inPoly = inpolygon(fineData.X,fineData.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
        fineData.X(~inPoly) = [];
        fineData.Y(~inPoly) = [];
        fineData.XComp(~inPoly,:) = [];
        fineData.YComp(~inPoly,:) = [];

        % concatenate fine model points to coarse model points
        data.X = [data.X; fineData.X];
        data.Y = [data.Y; fineData.Y];
        data.XComp = [data.XComp; fineData.XComp];
        data.YComp = [data.YComp; fineData.YComp];

    end

    % remove NaN points
    nanPoints = isnan(data.X);
    data.X(nanPoints) = [];
    data.Y(nanPoints) = [];
    data.XComp(nanPoints,:) = [];
    data.YComp(nanPoints,:) = [];
    clear nanPoints
    clear fineData

else % if not depth averaged (and therefore 3D)

    % load coarse grid horizontal velocity data at specified layer
    data = qpread(fidMain,'horizontal velocity','griddata',S.subset_t,S.subset_m,S.subset_n,S.d3d_verticalLayer);
    data.X = reshape(squeeze(data.X),[],1);
    data.Y = reshape(squeeze(data.Y),[],1);
    data.XComp = reshape(data.XComp,length(data.Time),[])';
    data.YComp = reshape(data.YComp,length(data.Time),[])';

    data.Name = ['horizontal_flow_velocity_verticalLayer_' num2str(S.d3d_verticalLayer,'%03f')];

    % add nested points if applicable
    if S.d3d_nested % remove coarse points where they overlap with nested finer domain
        inPoly = inpolygon(data.X,data.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
        data.X(inPoly) = []; % remove coarse domain points inside polygon
        data.Y(inPoly) = [];
        data.XComp(inPoly,:) = [];
        data.YComp(inPoly,:) = [];

        % load fine grid horizontal velocity data at specified layer
        fineData = qpread(fidFine,'horizontal velocity','griddata',S.subset_t,S.subset_m,S.subset_n);
        fineData.X = reshape(fineData.X,[],1);
        fineData.Y = reshape(fineData.Y,[],1);
        fineData.XComp = reshape(fineData.XComp,length(fineData.Time),[])';
        fineData.YComp = reshape(fineData.YComp,length(fineData.Time),[])';
        inPoly = inpolygon(fineData.X,fineData.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
        fineData.X(~inPoly) = []; % remove fine domain points outside polygon
        fineData.Y(~inPoly) = [];
        fineData.XComp(~inPoly,:) = [];
        fineData.YComp(~inPoly,:) = [];

        % concatenate fine model points to coarse model points
        data.X = [data.X; fineData.X];
        data.Y = [data.Y; fineData.Y];
        data.XComp = [data.XComp; fineData.XComp];
        data.YComp = [data.YComp; fineData.YComp];
        nanPoints = isnan(data.X); % remove NaN points
        data.X(nanPoints) = [];
        data.Y(nanPoints) = [];
        data.XComp(nanPoints,:) = [];
        data.YComp(nanPoints,:) = [];
        clear nanPoints
        clear fineData
    end

    nanPoints = isnan(data.X); % remove NaN points
    data.X(nanPoints) = [];
    data.Y(nanPoints) = [];
    data.XComp(nanPoints,:) = [];
    data.YComp(nanPoints,:) = [];
    clear nanPoints

end
%----------------------------------------------------------------------

%% Export current vectors
data.XComp;
data.YComp;

end