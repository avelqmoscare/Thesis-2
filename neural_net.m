function [] = neural_net()

global net;
global imds;
global trainingLables;
global augmentedTestSet;
global trainingFeatures;
global featureLayer;
global testSet;
global imageSize;

net.Layers(1);

%%
net.Layers(end);
numel(net.Layers(end).ClassNames);

%%
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
imageSize = net.Layers(1).InputSize;

%%
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet,'ColorPreprocessing', 'gray2rgb');

%%
% w1 = net.Layers(2).Weights;
% w1 = mat2gray(w1);

%%
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

%%

trainingLables = trainingSet.Labels;
end

