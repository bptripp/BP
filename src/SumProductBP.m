classdef SumProductBP < handle 
    %TODO
    % normalization
    % test against realistic stereo input with larger (downsampled) images
    %   (fewer levels)
        
    properties (Access = public)
        data = [];
        levels = [];
        messages = [];
        dataLogProb = [];
        h = [];
        dataSigma = [];
        discontinuitySigma = [];
        dt = .001;
        tau = .005;
    end
    
    methods (Access = public)
        function bp = SumProductBP(data, levels, dataSigma, discontinuitySigma)
            assert(ismatrix(data))
            
            bp.data = single(data);
            bp.levels = single(levels);
            bp.messages = ones(size(data,1), size(data,2), 4, length(levels), 'single') / length(levels);
            bp.messages = log(bp.messages); 
            bp.h = bp.messages;
            bp.dataSigma = dataSigma;
            bp.discontinuitySigma = discontinuitySigma;
            
            bp.dataLogProb = zeros(size(data,1), size(data,2), length(levels), 'single');
            for i = 1:size(data,1)
                for j = 1:size(data,2)
                    g = gaussian(bp.levels, bp.data(i,j), dataSigma) + .1/length(levels);
                    bp.dataLogProb(i,j,:) = log(g);
                    bp.dataLogProb(i,j,:) = rescaleLog(bp.dataLogProb(i,j,:));
                end
            end
            bp.dataLogProb = addNoiseAndBias(bp.dataLogProb);
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
            m = bp.dt/bp.tau * m + (1-bp.dt/bp.tau) * bp.messages; % continuous-time model
            bp.messages = addNoiseAndBias(m);
        end
        
        function message = getMessage(bp, row, col, neighbour)
            nl = length(bp.levels);
            
            % neighbour: 1 to 4 for left, right, above, below
            rowOffset = [0 0 -1 1];
            colOffset = [-1 1 0 0];
            inverseNeighbourList = [2 1 4 3];
            inverseNeighbour = inverseNeighbourList(neighbour);
            
            message = log(ones(size(bp.levels)) / nl); 
            
            fromRow = row + rowOffset(neighbour);
            fromCol = col + colOffset(neighbour);
            if fromRow >= 1 && fromRow <= size(bp.data,1) && fromCol >= 1 && fromCol <= size(bp.data,2)
                dataLogProb = squeeze(bp.dataLogProb(row, col, :)); 
                
                inMessages = squeeze(bp.messages(fromRow, fromCol, setdiff(1:4, inverseNeighbour), :));                
                messageLogProb = sum(inMessages,1); 
                
                lsp = dataLogProb' + messageLogProb;
                bp.h(row,col,neighbour,:) = bp.dt/bp.tau * lsp + (1-bp.dt/bp.tau) * squeeze(bp.h(row,col,neighbour,:))';
                logSourceProb = repmat(squeeze(bp.h(row,col,neighbour,:))', nl, 1);
                

                discontinuity = repmat(bp.levels, nl, 1) - repmat(bp.levels', 1, nl);
                discontinuityProb = gaussian(discontinuity, 0, bp.discontinuitySigma) + .1/nl;
                discontinuityProb = discontinuityProb ./ repmat(sum(discontinuityProb, 2), 1, nl); 
                logDiscontinuityProb = log(discontinuityProb);
                
                logProb = logSourceProb + logDiscontinuityProb;
                logProb = addNoiseAndBias(logProb + 8); %have to add something to avoid saturation
                
                message = log(sum(exp(logProb), 2)); 
                message = rescaleLog(message); 
            end
        end
        
        function result = getMAP(bp)
            prob = squeeze(sum(bp.messages,3)) + bp.dataLogProb; 
            [~,ind] = max(prob,[],3);
            result = bp.levels(ind);
        end
    end
end

function y = gaussian(x, mu, sigma)
    y = 1 / sigma / (2*pi)^.5 * exp(-(x-mu).^2 / 2 / sigma);
end

function logProb = rescaleLog(logProb)
    logProb = logProb - max(logProb);
end

function x = addNoiseAndBias(x)
    radius = 5; 
    x = (2*radius)*tanh(x/(2*radius)) + .2*radius*randn(size(x));
end
