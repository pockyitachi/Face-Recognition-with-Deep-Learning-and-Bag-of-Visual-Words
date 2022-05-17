clc;
clear all;
close all;
%% Read the dataset and split for training and testing
imds = imageDatastore('data', 'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);
[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomize');

%%
trainingSet.ReadFcn = @customReadDatastoreImage;
validationSet.ReadFcn = @customReadDatastoreImage;

%% Modified the K value for clusters
bag = bagOfFeatures(trainingSet,'TreeProperties',[1 500]);
%%
img = readimage(imds, 2);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure,
bar(featureVector)
title('500 Visual word')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

%%
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);


%%
confMatrix = evaluate(categoryClassifier, validationSet);
mean(diag(confMatrix))
%%
img = imread('testEL2D.JPG');
img = imresize(img,[224 224]);
figure
imshow(img)
[labelIdx, scores] = predict(categoryClassifier, img);
label = categoryClassifier.Labels(labelIdx);
hTitle = title(label);
