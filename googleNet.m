
clear all;
close all;
%% Prepare for the training and validation set
Dataset = imageDatastore('data', 'IncludeSubfolders',true,'LabelSource','foldernames');
[Training_Dataset, Validation_Dataset] = splitEachLabel(Dataset, 0.7,'randomized');

%% Load the original googlenet and show the layers
net = googlenet;
analyzeNetwork(net)

Input_Layer_Size = net.Layers(1).InputSize;

Layer_Graph = layerGraph(net);
%Layer 142 learn the feature of the recognition object
Feature_Learner = net.Layers(142);
Output_Classifier = net.Layers(144);
Number_of_Classes = numel(categories(Training_Dataset.Labels));

%% Build my own fully connected layer for replacing layer 142
New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Facial Feature Learner',...
    'WeightLearnRateFactor',10,...
    'BiasLearnRateFactor',10);

New_Classifier_Layer = classificationLayer('Name', 'Face Classifier');
 
Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);

Layer_Graph = replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);

analyzeNetwork(Layer_Graph)

%% Augument the dataset 
Pixel_Range = [-30 30];
Scale_Range = [0.9 1.1];
Image_Augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', Pixel_Range, ...
    'RandYTranslation', Pixel_Range, ...
    'RandXScale', Scale_Range, ...
    'RandYScale',Scale_Range);

Augmented_Training_Image = augmentedImageDatastore(Input_Layer_Size(1:2),Training_Dataset,...
    'DataAugmentation',Image_Augmenter);

Augmented_Validation_Image = augmentedImageDatastore(Input_Layer_Size(1:2),Validation_Dataset,...
    'DataAugmentation',Image_Augmenter);

%% Training
Size_of_Minibatch = 5;
Validation_Frequency = floor(numel(Augmented_Training_Image.Files)/Size_of_Minibatch);
Training_Options = trainingOptions('sgdm',...
    'MiniBatchSize',Size_of_Minibatch,...
    'MaxEpochs',10,...
    'InitialLearnRate',0.00003,...
    'Shuffle','Every-Epoch',...
    'ValidationData',Augmented_Validation_Image,...
    'ValidationFrequency',Validation_Frequency,...
    'Verbose',false,...
    'Plots','training-progress');
Trained_net = trainNetwork(Augmented_Training_Image, Layer_Graph, Training_Options);

%% Testing
I = imread('testP.jpg');
I = imresize(I,[224, 224]);
[Label, Prob] = classify(Trained_net, I);
if max(Prob) < 0.7
    Label = "unknown person";
end
figure,
imshow(I);
title({char(Label), num2str(max(Prob), 2)});
