function [net] = res_train_classes()
%%

outputFolder = fullfile('Classes');
rootFolder = fullfile(outputFolder, 'verify');

%%
categories = {'A', 'B', 'other'};
global imds;
imds = imageDatastore(fullfile(rootFolder,categories), 'LabelSource', 'foldernames');

%%
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

%%
imds = splitEachLabel(imds, minSetCount,'randomize');
countEachLabel(imds);

%%
A = find(imds.Labels == 'A', 1);
B = find(imds.Labels == 'B', 1);
other = find(imds.Labels == 'other', 1);
%%
global net;
net = resnet50();

end


