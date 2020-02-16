function [labels] = classify_model(img)

global augmentedTestSet;
global trainingFeatures;
global trainingLables;
global net;
global featureLayer;
global testSet;
global imageSize;
global classifier;


%%
% testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
% predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
% 
% %%
% testLables = testSet.Labels;
% confMat = confusionmat(testLables, predictLabels);
% 
% %%
% confMat = bsxfun(@rdivide, confMat, sum(confMat,2));
% mean(diag(confMat));

newImage = img;
%%
ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

%%
labels = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
end

