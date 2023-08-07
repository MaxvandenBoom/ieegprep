%
% Script to visualize the performances of the Brainvision reader (test_fileio_bv_perf.py) and EDF reader (test_fileio_edf_perf.py) unit-tests
% To specify which results to visualize, adjust the line below to pick a specific .mat file
%
load('D:\BIDS_erdetect\fileio_results\results_bv_win7_multiplexed__20230213_105805__200.mat');


%%

close all
figure('position', [0 200 1500 300]);

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

test_names = fieldnames(tests);

%for iTest = 3:3
for iTest = 1:length(test_names)
    subplot(1, length(test_names), iTest);
    test = tests.(test_names{iTest});
    
      
    % memmap & chunked
    memmap_chunked_idx     = find(test.conditions(:, 1) == 1 & test.conditions(:, 2) == 1, 1);
    memmap_nonchunked_idx  = find(test.conditions(:, 1) == 1 & test.conditions(:, 2) == 0, 1);
    stdio_chunked_idx      = find(test.conditions(:, 1) == 0 & test.conditions(:, 2) == 1, 1);
    stdio_nonchunked_idx   = find(test.conditions(:, 1) == 0 & test.conditions(:, 2) == 0, 1);
    
    xTickLabels = {};
    
    %
    % un-cached
    %
    yyaxis left
    
    xIndex = 1;
    plotViolin(xIndex, test.cond_results_uncached{memmap_chunked_idx, 4}, [.7 .8 1]);
    xTickLabels = [xTickLabels, 'ch mem'];
    
    xIndex = xIndex  + 1;
    plotViolin(xIndex, test.cond_results_uncached{memmap_nonchunked_idx, 4}, [.7 .8 1]);
    xTickLabels = [xTickLabels, 'noch mem'];
    
    xIndex = xIndex  + 1;
    plotViolin(xIndex, test.cond_results_uncached{stdio_chunked_idx, 4}, [.7 .8 1]);
    xTickLabels = [xTickLabels, 'ch io'];

    if ~isempty(stdio_nonchunked_idx)
        xIndex = xIndex  + 1;
        plotViolin(xIndex, test.cond_results_uncached{stdio_nonchunked_idx, 4}, [.7 .8 1]);
        xTickLabels = [xTickLabels, 'noch io'];
    end
    if iTest == 1
       ylabel('Un-cached') 
    end
    
    xIndex = xIndex + 1;
    xIndex = xIndex + 1;
    xTickLabels = [xTickLabels, ' '];
    xTickLabels = [xTickLabels, ' '];
    
    
    %
    % cached
    %
    yyaxis right
    
    xIndex = xIndex + 1;
    plotViolin(xIndex, test.cond_results_cached{memmap_chunked_idx, 4}, [1 .8 .7]);
    xTickLabels = [xTickLabels, 'ch mem'];
    
    %
    xIndex = xIndex  + 1;
    plotViolin(xIndex, test.cond_results_cached{memmap_nonchunked_idx, 4}, [1 .8 .7]);
    xTickLabels = [xTickLabels, 'noch mem'];
    
    
    xIndex = xIndex  + 1;
    plotViolin(xIndex, test.cond_results_cached{stdio_chunked_idx, 4}, [1 .8 .7]);
    xTickLabels = [xTickLabels, 'cu io'];

    if ~isempty(stdio_nonchunked_idx)
        xIndex = xIndex  + 1;
        plotViolin(xIndex, test.cond_results_cached{stdio_nonchunked_idx, 4}, [1 .8 .7]);
        xTickLabels = [xTickLabels, 'noch io'];
    end
    
    if iTest == length(test_names)
       ylabel('Cached') 
    end
    
    
    xticks([1:length(xTickLabels)]);
    xticklabels(strrep(xTickLabels, '_', '\_'));
    xtickangle(90)
    xlim([0 xIndex + 1]);
    
    
    %ytickformat('%.2f')
    ax=gca;
    ax.YAxis(1).Exponent = 0;
    ax.YAxis(2).Exponent = 0;
    title(strrep(test_names{iTest}, '_', '\_'));
end

function plotViolin(xIndex, values, color)
    values = rmoutliers(values, 'quartiles');
    
    violinPlot = Violin(values, xIndex, ...
							'Width', 0.4, ...
                            'ViolinColor', color, ...
                            'EdgeColor', color, ...
                            'ShowData', false, ...
                            'ViolinAlpha', 0.1, ...
                            'BoxWidth', 0.05, ...
                            'BoxColor', color, ...
                            'MedianColor', [0 0 0]);
    
    % mark mean
    scoreMean = mean(values);
    plot([xIndex - 0.3, xIndex + 0.3], [scoreMean, scoreMean], '-', 'LineWidth', 2, 'Color', 'k');

    % remove the dot for the median, we plot horizontal lines instead
    medianPlot = violinPlot.MedianPlot;
    set(medianPlot, 'SizeData', 3);
    delete(medianPlot);

    % 
    boxPlot = violinPlot.BoxPlot;
    set(boxPlot, 'facealpha', .5);
    
    
end
