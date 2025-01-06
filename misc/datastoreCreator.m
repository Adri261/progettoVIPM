function [trainds, unlabledds, valds, degds] = datastoreCreator()
    trainSmall = readtable('..\train_small.csv ');
    trainUnlabled = readtable('..\train_unlabeled.csv');
    val = readtable('..\val_info.csv');

    labledTrainFiles = trainSmall{:,1};
    labledTrainLables = trainSmall{:,2};

    unlabledTrainFiles = trainUnlabled{:,1};

    valFiles = val{:,1};
    valLables = val{:,2};
    % TODO: CAMBIARE I PATH!!!
    trainSetFile = "C:\Users\marco\Desktop\Uni\Magistrale\VIPM\Progetto\progettoVIPM\train_set";
    valSetFileNoDeg = "C:\Users\marco\Desktop\Uni\Magistrale\VIPM\Progetto\progettoVIPM\val_set";
    valSetFileDeg = "C:\Users\marco\Desktop\Uni\Magistrale\VIPM\Progetto\progettoVIPM\val_set_degraded";

    labledTrainSetPath = fullfile(trainSetFile, labledTrainFiles);
    unlabledTrainSetPath = fullfile(trainSetFile, unlabledTrainFiles);
    valSetNoDegPath = fullfile(valSetFileNoDeg, valFiles);
    valSetDegPath = fullfile(valSetFileDeg, valFiles);

    trainds = imageDatastore(labledTrainSetPath, IncludeSubfolders=true);
    unlabledds = imageDatastore(unlabledTrainSetPath,"IncludeSubfolders",true);
    valds = imageDatastore(valSetNoDegPath,"IncludeSubfolders",true);
    degds = imageDatastore(valSetDegPath,"IncludeSubfolders",true);
    trainds.Labels = categorical(labledTrainLables);
    valds.Labels = categorical(valLables);
    degds.Labels = categorical(valLables);
end