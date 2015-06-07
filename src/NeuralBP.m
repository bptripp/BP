classdef NeuralBP < handle 
    % Neural model of loopy belief propagation. We treat data and
    % dataLogProb as abstract inputs and encode h and messages with
    % neurons.
    
    properties (Access = public)
        data = [];
        levels = [];
        messages = [];
        dataLogProb = [];
        h = [];
        dataSigma = [];
        discontinuitySigma = [];
        dt = .0005;
        tau = .005;
              
        nPerState = 100;
        flatInd = []; % indices of state variables 
        hSpikeGen = [];
        hEncoders = [];
        hDecoders = [];
        messageSpikeGen = [];
        messageEncoders = [];
        messageDecoders = [];
        
        probedMessages = [];
    end
    
    methods (Access = public)
        function bp = NeuralBP(data, levels, dataSigma, discontinuitySigma)
            assert(ismatrix(data))
            
            bp.data = single(data);
            bp.levels = single(levels);
            bp.messages = ones(size(data,1), size(data,2), 4, length(levels), 'single') / length(levels);
            bp.messages = log(bp.messages); 
            bp.probedMessages = bp.messages;
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
            
            bp.flatInd = reshape(1:numel(bp.h), size(bp.h));
            
            tauRef = .002;
            tauRC = .02;
            radius = 5;            
            
            hIntercepts = -1 + 2*rand(1, bp.nPerState*numel(bp.flatInd));
            hMaxRates = 100 + 100*rand(1, bp.nPerState*numel(bp.flatInd));
            bp.hSpikeGen = LIFSpikeGenerator(bp.dt, tauRef, tauRC, hIntercepts, hMaxRates, 0);
            bp.hEncoders = -1 + 2*(rand(1,bp.hSpikeGen.n)<.5);
%             bp.hDecoders = findDecoders(bp.hSpikeGen, bp.hEncoders, radius, bp.nPerState, @(x) x);

            messageIntercepts = -1 + 2*rand(1, bp.nPerState*numel(bp.flatInd));
            messageMaxRates = 100 + 100*rand(1, bp.nPerState*numel(bp.flatInd));
            bp.messageSpikeGen = LIFSpikeGenerator(bp.dt, tauRef, tauRC, messageIntercepts, messageMaxRates, 0);  
            bp.messageEncoders = -1 + 2*(rand(1,bp.messageSpikeGen.n)<.5);
            bp.messageDecoders = findDecoders(bp.messageSpikeGen, bp.messageEncoders, radius, bp.nPerState, @(x) x);
        end
        
        function iterate(bp, startTime)
            m = zeros(size(bp.messages), 'single');
            for i = 1:size(bp.data, 1)
                for j = 1:size(bp.data, 2)
                    for k = 1:4
                        m(i, j, k, :) = getMessage(bp, i, j, k);
                    end
                end
            end            
            bp.messages = bp.dt/bp.tau * m + (1-bp.dt/bp.tau) * bp.messages; % input to message neurons
            
            drive = bp.messageEncoders .* reshape(ones(bp.nPerState,1) * reshape(bp.messages, 1, []), 1, []);
            spikes = bp.messageSpikeGen.run(drive', startTime, startTime+bp.dt, 1);
            decoded = sum(reshape(bp.messageDecoders .* spikes', bp.nPerState, []), 1);
            bp.probedMessages =  bp.dt/bp.tau * reshape(decoded, size(bp.messages)) + (1-bp.dt/bp.tau) * bp.probedMessages;
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
                discontinuityProb = gaussian(discontinuity, 0, bp.discontinuitySigma) + .05/nl;
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
        
        function result = getProbeMAP(bp)
            prob = squeeze(sum(bp.probedMessages,3)) + bp.dataLogProb; 
            [~,ind] = max(prob,[],3);
            result = bp.levels(ind);
        end
    end
end

function result = decodeFlatSpikes(spikes, decoders, nPerState)
    % spikes: a vector in which the first nPerState values belong to the 
    %   first state variable, etc. 
    % decoders: same shape as spikes (one decoder per neuron)
    % nPerState: number of neurons per state variable
    product = reshape(spikes .* decoders, nPerState, []);
    result = sum(product, 1);
end

function result = findDecoders(spikeGenerator, encoders, radius, nPerState, f)
    x = -radius:radius/100:radius;
    ideal = f(x);
    
    drive = encoders' * x;
    rates = spikeGenerator.getRates(drive, 0, 0);
    
    nStates = spikeGenerator.n / nPerState;
    result = zeros(1, spikeGenerator.n);
    relNoise = .1;
    for i = 1:nStates
        i
        ind = (i-1)*nPerState + (1:nPerState);
        r = rates(ind,:);
        gamma = r * r';
        gamma = gamma + (relNoise*max(r(:)))^2*size(gamma,1) * eye(size(gamma,1));
%         invgamma = pinv(gamma);
        V = r * ideal';
        d = (gamma\V)';
        result(ind) = d;
    end
end

function y = gaussian(x, mu, sigma)
    y = 1 / sigma / (2*pi)^.5 * exp(-(x-mu).^2 / 2 / sigma);
end

function logProb = rescaleLog(logProb)
    %TODO: get rid of this
    logProb = logProb - max(logProb);
end

function x = addNoiseAndBias(x)
    radius = 5; 
    x = (2*radius)*tanh(x/(2*radius)) + .1*radius*randn(size(x));
end
