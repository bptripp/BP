function cnn = cnnaddneurons(cnn, layer, map, n)
    assert(strcmp('c', cnn.layers{layer}.type)) %convolutional layer
    
    if ~isfield(cnn.layers{layer}, 'encoders')
        cnn.layers{layer}.encoders = cell(1,cnn.layers{layer}.outputmaps);
    end
    
    if ~isfield(cnn.layers{layer}, 'spikegenerators')
        cnn.layers{layer}.spikegenerators = cell(1,cnn.layers{layer}.outputmaps);
    end
    
    kernelSize = size(cnn.layers{layer}.k{1}{1});
    inputSize = size(cnn.layers{layer-1}.a{1});    
    s = inputSize(1:2) - kernelSize + 1;
    nUnits = prod(s);
    
    tauRef = .002;
    tauRC = .02;
    intercepts = -1 + 2*rand(1, n);
    maxRates = 100 + 100*rand(1, n); 
    cnn.layers{layer}.spikegenerators{map} = LIFSpikeGenerator(.001, tauRef, tauRC, intercepts, maxRates, 0);
    
    encoders = randn(nUnits, n);
    for i = 1:n %normalize encoders and make them spatially localized
        centre = [1 1] + [(s(1)-1)*rand (s(2)-1)*rand];
        dist = repmat(((1:s(1))'-centre(1)).^2, 1, s(2)) + repmat(((1:s(2))-centre(2)).^2, s(1), 1);
        sigma = 1;
        scale = exp(-dist/2/sigma);
        encoders(:,i) = encoders(:,i) .* scale(:);
        encoders(:,i) = encoders(:,i) ./ norm(encoders(:,i));
    end    
    cnn.layers{layer}.encoders{map} = encoders;
end
