
% Crea le rappresentazioni BOW dei dataframe Train e Val

function [BOWTR, labelsTR, BOWVAL, labelsVAL, C, IDX] = BOWCreator(Train,Val,featStep,imsize,featureFuncs,K)
if isempty(featStep)
    featStep = 5;
end
if isempty(imsize)
    imsize = 200;
end
if isempty(featureFuncs)
    featureFuncs = ['SURF'];
end
if isempty(K)
    K = 100;
end
%% estrazione features training
[ii, jj] = meshgrid(featStep:featStep:imsize-featStep, featStep:featStep:imsize-featStep);
pointPositions = [ii(:), jj(:)];
features=[];
labels=[];
disp("Estrazione Features Training")
for idx = 1:numel(Train.Files)
    im = readimage(Train,idx);
    class = Train.Labels(idx);
    im = im2double(im);
    im = imresize(im,[imsize imsize]);
    im = rgb2gray(im);
    imageFeatures = [];


    for func = featureFuncs
        [extractedFeature,~] = extractFeatures(im,pointPositions,'Method',func);
        if isstruct(extractedFeature)
            extractedFeature = extractedFeature.Features;
        end
        imageFeatures = [imageFeatures, extractedFeature(:)'];
    end
    features = [features; imageFeatures];
    labels = [labels;categorical(repmat(class,size(imageFeatures,1),1)),categorical(repmat(idx,size(imageFeatures,1),1))];
end
save("tmp.mat")
%% creazione vocabolario
disp("Creazione vocabolario")
[IDX,C]=kmeans(features,K);
save("tmp.mat")

%% istogrammi training

disp("Creazione istogrammi train")
BOW_tr = [];
labels_tr = [];
for idx = 1:numel(Train.Files)
    class = Train.Labels(idx);
    u=labels(:,1)==categorical(class) & labels(:,2)==categorical(idx);
    imfeaturesIDX=IDX(u);
    H=hist(imfeaturesIDX,1:K);
    H=H/sum(H);
    BOW_tr=[BOW_tr; H];
    labels_tr=[labels_tr; class];
end
save("tmp.mat")
%% istogrammi validation
BOW_va = [];
labels_va = [];
disp("Creazione istogrammi val")
for idx = 1:numel(Val.Files)
    class = Val.Labels(idx);
    im = im2double(im);
    im = imresize(im,[imsize imsize]);
    im = im2gray(im);
    imageFeatures = [];
    for func = featureFuncs
        [extractedFeature,~] = extractFeatures(im,pointPositions,'Method',func);
        if isstruct(extractedFeature)
            extractedFeature = extractedFeature.Features;
        end
        imageFeatures = [imageFeatures, extractedFeature(:)'];
    end
    D = pdist2(imageFeatures, C);
    [~,words] = min(D,[],2);
    H = hist(words,1:K);
    H = H./sum(H);
    BOW_va = [BOW_va;H];
    labels_va = [labels_va;class];
end
save("tmp.mat")
%% return
BOWTR = BOW_tr;
labelsTR = labels_tr;
BOWVAL = BOW_va;
labelsVAL = labels_va;

end