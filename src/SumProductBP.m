classdef SumProductBP < handle 
    %TODO
    % try product in place of min
    % test against realistic stereo input with larger (downsampled) images
    %   (fewer levels)
        
    properties (Access = public)
        data = [];
        levels = [];
        messages = [];
        dataProb = [];
        dataSigma = [];
        discontinuitySigma = [];
    end
    
    methods (Access = public)
        function bp = SumProductBP(data, levels, dataSigma, discontinuitySigma)
            assert(ismatrix(data))
            
            bp.data = single(data);
            bp.levels = single(levels);
            bp.messages = ones(size(data,1), size(data,2), 4, length(levels), 'single') / length(levels);
            bp.dataSigma = dataSigma;
            bp.discontinuitySigma = discontinuitySigma;
            
            bp.dataProb = zeros(size(data,1), size(data,2), length(levels), 'single');
            for i = 1:size(data,1)
                for j = 1:size(data,2)
                    bp.dataProb(i,j,:) = gaussian(bp.levels, bp.data(i,j), dataSigma) + .1 / length(levels); 
                    bp.dataProb(i,j,:) = bp.dataProb(i,j,:) ./ sum(bp.dataProb(i,j,:)); % normalize
                end
            end            
        end
        
        function iterate(bp)
            m = zeros(size(bp.messages), 'single');
            for i = 1:size(bp.data, 1)
                for j = 1:size(bp.data, 2)
                    for k = 1:4
                        m(i, j, k, :) = getMessage(bp, i, j, k);
                    end
                end
            end
            bp.messages = m;
        end
        
        function message = getMessage(bp, row, col, neighbour)
            nl = length(bp.levels);
            
            % neighbour: 1 to 4 for left, right, above, below
            rowOffset = [0 0 -1 1];
            colOffset = [-1 1 0 0];
            inverseNeighbourList = [2 1 4 3];
            inverseNeighbour = inverseNeighbourList(neighbour);
            
            message = ones(size(bp.levels)) / nl;
            
            fromRow = row + rowOffset(neighbour);
            fromCol = col + colOffset(neighbour);
            if fromRow >= 1 && fromRow <= size(bp.data,1) && fromCol >= 1 && fromCol <= size(bp.data,2)
                dataProb = squeeze(bp.dataProb(row, col, :)); 
                
                inMessages = squeeze(bp.messages(fromRow, fromCol, setdiff(1:4, inverseNeighbour), :));                
                messageProb = prod(inMessages,1); 
                
                sourceProb = repmat(dataProb' .* messageProb, nl, 1);

                discontinuity = repmat(bp.levels, nl, 1) - repmat(bp.levels', 1, nl);
                discontinuityProb = gaussian(discontinuity, 0, bp.discontinuitySigma) + .1/nl;
                discontinuityProb = discontinuityProb ./ repmat(sum(discontinuityProb, 2), 1, nl); 
                
                prob = sourceProb .* discontinuityProb;
                
                message = sum(prob, 2);
                message = message / sum(message);
            end
        end
        
        function result = getMAP(bp)
            prob = squeeze(prod(bp.messages,3)) .* bp.dataProb; 
            [~,ind] = max(prob,[],3);
            result = bp.levels(ind);
        end
    end
end

function y = gaussian(x, mu, sigma)
    y = 1 / sigma / (2*pi)^.5 * exp(-(x-mu).^2 / 2 / sigma);
end

