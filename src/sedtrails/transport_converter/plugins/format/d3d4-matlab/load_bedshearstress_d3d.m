function [data] = load_bedshearstress_d3d(S,XYT)
% Load hydrodynamic data from d3d-4 model for sediment transport calculations

%% LOAD HYDRODYNAMIC VARIABLES

% specify main trim file to load
fidMain = qpfopen([S.d3d_path_mainTrimFile]);

%% MEAN BED SHEAR STRESS 
% N.B. mean bed shear stress is specified as a vector in Delft3D-4 but we
% use the magnitude in Soulsby 2011

% load main grid bed shear stress data and save as .mat file
data = qpread(fidMain,'bed shear stress','griddata',S.subset_t,S.subset_m,S.subset_n);
data.X = reshape(data.X,[],1);
data.Y = reshape(data.Y,[],1);
data.XComp = reshape(data.XComp,length(data.Time),[])';
data.YComp = reshape(data.YComp,length(data.Time),[])';

% add nested points if applicable
if S.d3d_nested 
    % remove coarse points where they overlap with nested finer domain
    inPoly = inpolygon(data.X,data.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
    data.X(inPoly) = []; % remove coarse domain points inside polygon
    data.Y(inPoly) = [];
    data.XComp(inPoly,:) = [];
    data.YComp(inPoly,:) = [];

    % load fine grid bed shear stress data
    fidFine = qpfopen([S.d3d_path_nestedFineTrimFile]);
    fineData = qpread(fidFine,'bed shear stress','griddata',S.subset_t,S.subset_m,S.subset_n);
    fineData.X = reshape(fineData.X,[],1);
    fineData.Y = reshape(fineData.Y,[],1);
    fineData.XComp = reshape(fineData.XComp,length(fineData.Time),[])';
    fineData.YComp = reshape(fineData.YComp,length(fineData.Time),[])';
    inPoly = inpolygon(fineData.X,fineData.Y,xpol,ypol);
    fineData.X(~inPoly) = []; % remove fine domain points outside polygon
    fineData.Y(~inPoly) = [];
    fineData.XComp(~inPoly,:) = [];
    fineData.YComp(~inPoly,:) = [];

    % concatenate fine grid model points to coarse grid model points
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

% compute magnitude of mean bed shear stress
mean_bss_mag = sqrt(data.XComp.^2+data.YComp.^2);
clear data

%% MAXIMUM BED SHEAR STRESS 
% load main grid maximum bed shear stress data
data = qpread(fidMain,'maximum bed shear stress','griddata',S.subset_t,S.subset_m,S.subset_n);
data.X = reshape(data.X,[],1);
data.Y = reshape(data.Y,[],1);
data.Val = reshape(data.Val,length(data.Time),[])';

% add nested points if applicable
if S.d3d_nested 
    % remove coarse points where they overlap with nested finer domain
    inPoly = inpolygon(data.X,data.Y,S.d3d_nested_xpol,S.d3d_nested_ypol);
    data.X(inPoly) = []; % remove coarse domain points inside polygon
    data.Y(inPoly) = [];
    data.Val(inPoly,:) = [];

    % load fine grid maximum bed shear stress data
    fidFine = qpfopen([S.d3d_path_nestedFineTrimFile]);
    fineData = qpread(fidFine,'maximum bed shear stress','griddata',S.subset_t,S.subset_m,S.subset_n);
    fineData.X = reshape(fineData.X,[],1);
    fineData.Y = reshape(fineData.Y,[],1);
    fineData.Val = reshape(fineData.Val,[],length(fineData.Time));
    inPoly = inpolygon(fineData.X,fineData.Y,xpol,ypol);
    fineData.X(~inPoly) = []; % remove fine domain points outside polygon
    fineData.Y(~inPoly) = [];
    fineData.Val(~inPoly,:) = [];

    % concatenate fine model points to coarse model points
    data.X = [data.X; fineData.X];
    data.Y = [data.Y; fineData.Y];
    data.Val = [data.Val; fineData.Val];
    nanPoints = isnan(data.X); % remove NaN points
    data.X(nanPoints) = [];
    data.Y(nanPoints) = [];
    data.Val(nanPoints,:) = [];
    clear nanPoints
    clear fineData
end

nanPoints = isnan(data.X); % remove NaN points
data.X(nanPoints) = [];
data.Y(nanPoints) = [];
data.Val(nanPoints,:) = [];
clear nanPoints

%% EXPORT DATA
data.mean_bss_mag = mean_bss_mag;
data.max_bss_mag = data.Val;
data = rmfield(data,'Val');

end