classdef BP < handle 
    %TODO
    % normalize everything to between 0 and 1 (for neurons and log)
    % try product in place of min
    % test against realistic stereo input with larger (downsampled) images
    %   (fewer levels)
    
    
    properties (Access = public)
        data = [];
        levels = [];
        messages = [];
        dataCost = [];
    end
    
    methods (Access = public)
        function bp = BP(data, levels)
            assert(ismatrix(data))
            assert(isa(data, 'int32'))
            assert(isa(levels, 'int32'))
            
            bp.data = single(data);
            bp.levels = single(levels);
            bp.messages = zeros(size(data,1), size(data,2), 4, length(levels), 'single');
            
            maxDataCost = 10;
            
            bp.dataCost = zeros(size(data,1), size(data,2), length(levels), 'single');
            for i = 1:size(data,1)
                for j = 1:size(data,2)
                    bp.dataCost(i,j,:) = min(maxDataCost, abs(bp.levels - bp.data(i,j))); 
                end
            end
            
%             bp.dataCost = bp.dataCost / maxDataCost; % normalize to max 1
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
            % neighbour: 1 to 4 for left, right, above, below
            rowOffset = [0 0 -1 1];
            colOffset = [-1 1 0 0];
            inverseNeighbourList = [2 1 4 3];
            inverseNeighbour = inverseNeighbourList(neighbour);
            
            maxDiscontinuityCost = 10;
            
            fromRow = row + rowOffset(neighbour);
            fromCol = col + colOffset(neighbour);
            message = zeros(size(bp.levels));
            if fromRow >= 1 && fromRow <= size(bp.data,1) && fromCol >= 1 && fromCol <= size(bp.data,2)
                dc = squeeze(bp.dataCost(row, col, :)); 
                
                inMessages = squeeze(bp.messages(fromRow, fromCol, setdiff(1:4, inverseNeighbour), :));
                
                messageCost = sum(inMessages,1); 
                sourceCost = repmat(dc' + messageCost, length(bp.levels), 1);
                
%                 sourceCost = sourceCost / 4;
                
                discontinuityCost = min(maxDiscontinuityCost, abs(repmat(bp.levels, length(bp.levels), 1) - repmat(bp.levels', 1, length(bp.levels))));
%                 discontinuityCost = discontinuityCost / maxDiscontinuityCost; % normalize to max 1
                
                cost = sourceCost + discontinuityCost;
%                 cost = sourceCost/2 + discontinuityCost/2;
                
                if row == 5 && col == 5 && neighbour == 1
                    figure(2), set(gcf, 'Position', [782   374   560   420])
                    subplot(1,3,1), mesh(sourceCost), subplot(1,3,2), mesh(discontinuityCost), subplot(1,3,3), mesh(cost)
                end
                
                message = min(cost, [], 2);
                message = message - sum(message) / length(message); % blows up otherwise
            end
        end
        
        function result = getMAP(bp)
            cost = squeeze(sum(bp.messages,3)) + bp.dataCost; 
            [~,ind] = min(cost,[],3);
            result = bp.levels(ind);
        end
        
        
        % Below are functions ported from Felzenszwalb's code ------------
        % TODO: not done porting (doesn't run)
        
        function f = dt(f, values)
            % Distance transform of 1d function
            % 
            % f: the function
            % values: number of disparities
            for q = 2:values
                prev = f(q-1) + 1;
                if prev < f(q)
                    f(q) = prev;
                end
            end
            for q = values-1:-1:1
                prev = f(q+1) + 1;
                if (prev < f(q))
                    f(q) = prev;
                end
            end
        end
        
        function dst = msg(s1, s2, s3, s4, values)
            % values: number of disparities
            DISC_K = 1.7; % truncation of discontinuity cost 

            % aggregate and find min
            dst = s1 + s2 + s3 + s4;
            minimum = min(dst);

            dt(dst);

            % truncate 
            minimum = minimum + DISC_K;
            dst(minimum < dst) = minimum;

            % normalize
            dst = dst - sum(dst) / values; 
        end
        
        % computation of data costs
        function data = comp_data(img1, img2, values)
            SIGMA = 0.7;    % amount to smooth the input images
            DATA_K = 15.0;  % truncation of data cost
            LAMBDA = 0.07;  % weighting of data cost

            width = size(img1,2);
            height = size(img1, 1);
            data = zeros(height, width);

            if SIGMA >= 0.1
                SS = ceil(4*SIGMA);
                H = 1 / (2*pi)^.5 / SIGMA * exp(-(-SS:SS)/2/SIGMA^2);
                sm1 = conv2(H, H, img1, 'same');
                sm2 = conv2(H, H, img2, 'same');
            else 
                sm1 = img1;
                sm2 = img2;
            end

            data = zeros(size(img1,1), size(img1,2)-values, values);
            for value = 1:values
                data(:,:,value) = sm1(values+1:end) - sm2(1:end-values);
            end
            data = LAMBDA * min(data, DATA_K);            
        end

        function out = output(u, d, l, r, data, VALUES) 
            % generate output from current messages
            
            width = size(data,2);
            height = size(data,1);
            out = zeros(height, width);

            for y = 1:height
                for x = 1:width
                    % keep track of best value for current pixel
                    best = 0;
                    best_val = Inf;
                    for value = 0:VALUES
                        val = u(x, y+1, value) + d(x, y-1, value) + l(x+1, y, value) + r(x-1, y, value) + data(x, y, value);
                        if val < best_val 
                            best_val = val;
                            best = value;
                        end
                    end
                end
            end
            out = best;
        end

        function bp_cb(u, d, l, r, data, iter)
            % belief propagation using checkerboard update scheme
            % 
            % iter: number of iterations

            width = size(data, 2);  
            height = size(data, 1);

            for t = 0:iter
                disp(['iter' t])

                %TODO: are edges skipped?
                for y = 2:height-1
                    for x = mod(y+t,2)+1:2:width-1

                        %msg(s1, s2, s3, s4, values)
                        msg(imRef(u, x, y+1),imRef(l, x+1, y),imRef(r, x-1, y),imRef(data, x, y), imRef(u, x, y));

                        msg(imRef(d, x, y-1),imRef(l, x+1, y),imRef(r, x-1, y),imRef(data, x, y), imRef(d, x, y));

                        msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(r, x-1, y),imRef(data, x, y), imRef(r, x, y));

                        msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(l, x+1, y),imRef(data, x, y), imRef(l, x, y));
                    end
                end
            end
        end

        % -----------------------------------------------------------------
        
    end
    
end

