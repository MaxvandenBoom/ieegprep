%
% Script to visualize the performances of the different epoch routines on the file format readers
%

%{
% epoch
load('D:\BIDS_erdetect\fileio_results\results_epoch_win7__20230726_080359.mat');
test_formats = {'BrainVision (multiplexed)', 'bv_multiplexed'; ...
                'BrainVision (vectorized)', 'bv_vectorized'; ...
                'EDF', 'edf'; ...
                'MEF3', 'mef'};
test_functions = {'by_channels', 'by_trials'};
%}


% epoch average
load('D:\BIDS_erdetect\fileio_results\results_epochAverage_win7__20230807_054418.mat');
test_formats = {'BrainVision (multiplexed)', 'bv_mult'; ...
                'BrainVision (vectorized)', 'bv_vect'; ...
                'EDF', 'edf'; ...
                'MEF3', 'mef'};
test_functions = {'by_ch_cond_trial', 'by_cond_trials', 'by_prep_mem', 'by_prep_speed'};


%%

close all
figure('position', [0 200 1500 700]);

% remove edges
%{
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
%}


% loop over the different data-formats (one graph per format)
for iFormat = 1:size(test_formats, 1)
    subplot(1, size(test_formats, 1), iFormat);
    format_short = test_formats{iFormat, 2};
    
    xTickLabels = {};
    
    uncachedPrepColor = [0.6667, 0.8235, 0.9804];
    uncachedEpochColor = [0.0784, 0.4706, 0.8039];
    
    cachedPrepColor = [1.0000, 0.6667, 0.6667];
    cachedEpochColor = [1.0000, 0.2353, 0.2353];
    
    %
    % un-cached
    %
    yyaxis left
    xIndex = 1;
    maxY = 0;
    
    for iFunc = 1:length(test_functions)
        test = tests.([test_functions{iFunc}, '__', format_short]);
        nopreload_idx     = find(test.conditions == 0, 1);
        preload_idx       = find(test.conditions == 1, 1);
        
        
        
        % no preload - by ...
        pMaxY = plotBar(xIndex, test.cond_epoch_results_uncached{nopreload_idx, 4} / 1000, uncachedEpochColor, ...
                        test.cond_prep_results_uncached{nopreload_idx, 1} / 1000, 2, '-');
        maxY = max(maxY, pMaxY);
        plotBar(xIndex, test.cond_prep_results_uncached{nopreload_idx, 4} / 1000, uncachedPrepColor, 0, 1, '-');
        xTickLabels = [xTickLabels, ['no_preload - ', test_functions{iFunc}]];
        xIndex = xIndex  + 1;

    end
    
    xIndex = xIndex  + 2;
    xTickLabels = [xTickLabels, ' '];
    xTickLabels = [xTickLabels, ' '];
    
    for iFunc = 1:length(test_functions)
        test = tests.([test_functions{iFunc}, '__', format_short]);
        nopreload_idx     = find(test.conditions == 0, 1);
        preload_idx       = find(test.conditions == 1, 1);

        % preload - by ...
        pMaxY = plotBar(xIndex, test.cond_epoch_results_uncached{preload_idx, 4} / 1000, uncachedEpochColor, ...
                        test.cond_prep_results_uncached{preload_idx, 1} / 1000, 2, '-');
        maxY = max(maxY, pMaxY);
        plotBar(xIndex, test.cond_prep_results_uncached{preload_idx, 4} / 1000, uncachedPrepColor, 0, 1, '-');
        xTickLabels = [xTickLabels, ['preload - ', test_functions{iFunc}]];
        xIndex = xIndex  + 1;
        
    end
    
    if iFormat == 1
       ylabel('Un-cached') 
    end
    
    xIndex = xIndex + 2;
    xTickLabels = [xTickLabels, ' '];
    xTickLabels = [xTickLabels, ' '];
    ylim([0 maxY]);
    
    
    %
    % cached
    %
    yyaxis right
    maxY = 0;
    
    for iFunc = 1:length(test_functions)
        test = tests.([test_functions{iFunc}, '__', format_short]);
        nopreload_idx     = find(test.conditions == 0, 1);
        preload_idx       = find(test.conditions == 1, 1);
        
        % no preload - by ...
        pMaxY = plotBar(xIndex, test.cond_epoch_results_cached{nopreload_idx, 4} / 1000, cachedEpochColor, ...
                        test.cond_prep_results_cached{nopreload_idx, 1} / 1000, 2, '-');
        maxY = max(maxY, pMaxY);
        plotBar(xIndex, test.cond_prep_results_cached{nopreload_idx, 4} / 1000, cachedPrepColor, 0, 1, '-');
        xTickLabels = [xTickLabels, ['no_preload - ', test_functions{iFunc}]];
        xIndex = xIndex + 1;
        
    end
    
    xIndex = xIndex + 2;
    xTickLabels = [xTickLabels, ' '];
    xTickLabels = [xTickLabels, ' '];
    
    for iFunc = 1:length(test_functions)
        test = tests.([test_functions{iFunc}, '__', format_short]);
        nopreload_idx     = find(test.conditions == 0, 1);
        preload_idx       = find(test.conditions == 1, 1);
        
        % preload - by ...
        pMaxY = plotBar(xIndex, test.cond_epoch_results_cached{preload_idx, 4} / 1000, cachedEpochColor, ...
                        test.cond_prep_results_cached{preload_idx, 1} / 1000, 2, '-');
        maxY = max(maxY, pMaxY);
        plotBar(xIndex, test.cond_prep_results_cached{preload_idx, 4} / 1000, cachedPrepColor, 0, 1, '-');
        xTickLabels = [xTickLabels, ['preload - ', test_functions{iFunc}]];
        xIndex = xIndex + 1;
        
    end
    
    if iFormat == size(test_formats, 1)
       ylabel('Cached') 
    end
    
    xticks(1:length(xTickLabels));
    xticklabels(strrep(xTickLabels, '_', '\_'));
    xtickangle(90)
    xlim([0 xIndex]);
    ylim([0 maxY]);
    
    %ytickformat('%.2f')
    ax=gca;
    ax.YAxis(1).Exponent = 0;
    ax.YAxis(2).Exponent = 0;
    title(strrep(test_formats{iFormat, 1}, '_', '\_'));
end


function maxY = plotBar(xIndex, values, color, yOffset, lineWidth, LineStyle)
    values = rmoutliers(values, 'quartiles');

    scoreMean = mean(values);
    scoreStd = std(values);

    hold on;
    fill([xIndex - 0.25, xIndex + 0.25, xIndex + 0.25, xIndex - 0.25], ...
    [yOffset, yOffset, yOffset + scoreMean, yOffset + scoreMean], ...
    color, 'LineStyle', '-', 'Marker', 'None');

    % scatter
    %for iValue = 1:length(values)
    %    plot(xIndex - .2 + (rand * .4), yOffset + values(iValue), '.', 'Color', [.6 .6 .6]);
    %end
    
    % std error bars
    plot([xIndex, xIndex], [yOffset + scoreMean - scoreStd, yOffset + scoreMean + scoreStd], '-', 'LineWidth', 1, 'Color', 'k');
    plot([xIndex - 0.2, xIndex + 0.2], [yOffset + scoreMean + scoreStd, yOffset + scoreMean + scoreStd], '-', 'LineWidth', 1, 'Color', 'k');
    plot([xIndex - 0.2, xIndex + 0.2], [yOffset + scoreMean - scoreStd, yOffset + scoreMean - scoreStd], '-', 'LineWidth', 1, 'Color', 'k');
    maxY = yOffset + scoreMean + scoreStd;
    
    % mean
    plot([xIndex - 0.5, xIndex + 0.5], [yOffset + scoreMean, yOffset + scoreMean], LineStyle, 'LineWidth', lineWidth, 'Color', 'k');
    
    hold off;
end
