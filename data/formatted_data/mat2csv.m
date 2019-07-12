% Create data from desat struct
clear all;
clc;
%% Start by loading the .mat file

load('/home/buck06191/repos/Github/neonatal-desat-modelling/data/formatted_data/JCBFMData210917perbaby.mat');

% Data is stored in a struct called 'desats'

chosen_data = {'CYRIL 007 070114 224409 1 ALL COR.xlsx' 'CYRIL 021 250614 092941 1 ALL COR.xlsx'};

column_idx = [14, 33]; 

column_names= {'CO2' 'SpO2' 'MABP' 'HbD40' 'HbT40' 'CCO40'};

timings = desats.normtime;

for ii=1:2
    fprintf('%s\n', chosen_data{ii})
    fprintf(' %i \n %i\n\n', timings(1,column_idx(ii)), timings(3,column_idx(ii)));
end

%% Extract data from struct
neo007 = table();
neo021 = table();
for k=1:length(column_names)
    curname=column_names{k};
    neo007.(curname) = num2cell(desats.(curname)(:,14));
    neo021.(curname) = num2cell(desats.(curname)(:,33));
end

neo007.time = num2cell([0:1200]');
neo021.time = num2cell([0:1200]');

%% Write to file
cd([pwd,'/']);
writetable(neo007, 'neo007_mat.csv');
writetable(neo021, 'neo021_mat.csv');