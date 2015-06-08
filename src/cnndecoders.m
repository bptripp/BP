function [net, estimates] = cnndecoders(net, layer, map, rates, targets)

    % find decoders ... 
    gamma = rates * rates';
    relNoise = .1;
    gamma = gamma + (relNoise*max(rates(:)))^2*size(rates,2) * eye(size(gamma,1));
    V = rates * targets';
    decoders = (gamma\V)';
    
    % plot error ... 
    estimates = decoders * rates;
    error = estimates - targets;
    figure
    subplot(2,1,1), hist(targets(:), 100), set(gca, 'XLim', [-.04 .2]), title('target histogram')
    subplot(2,1,2), hist(error(:), 100), set(gca, 'XLim', [-.04 .2]), title('error histogram')
    
    % add to network ... 
    if ~isfield(net.layers{layer}, 'decoders')
        net.layers{layer}.decoders = cell(1,net.layers{layer}.outputmaps);
    end
    net.layers{layer}.decoders{map} = decoders;
end
