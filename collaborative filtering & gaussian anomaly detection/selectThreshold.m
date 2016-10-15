function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

	predictions = pval < epsilon;		% 1 == predicted anomaly
	
	true_positives = 0;					% accumulators for each measure
	false_positives = 0;
	false_negatives = 0;
	
	for i = 1:size(yval, 1)
		if predictions(i) == 1 && predictions(i) == yval(i)
			true_positives = true_positives + 1;
		end
		if predictions(i) == 1 && predictions(i) ~= yval(i)
			false_positives = false_positives + 1;
		end
		if predictions(i) == 0 && predictions(i) == 1
			false_negatives = false_negatives + 1;
		end
	end

	precision = true_positives / (false_positives + true_positives);		% computes necessary measures
	recall = true_positives / (false_negatives + true_positives);
	F1 = 2 * precision * recall / (precision + recall);

    % =============================================================

    if F1 > bestF1				% select best anomaly threshold
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
	
end

end
