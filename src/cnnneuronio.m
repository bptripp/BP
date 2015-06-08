function [states, targets, rates] = cnnneuronio(net, layer, map, x)
    % Adapted from cnnff.m by Palm. Samples states, spike rates, and target
    % outputs for a certain layer and map in a CNN. 
    % 
    % cnn: trained convolutional network 
    % layer: index of desired layer
    % map: index of desired feature map
    % x: input to first layer of network 
    
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : layer   %  for each layer up to given layer
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
                
                if l == layer && j == map
                    states = reshape(z, size(z,1)*size(z,2), []);
                    targets = reshape(net.layers{l}.a{j}, size(z,1)*size(z,2), []);
                end
                
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end
    
    encoders = net.layers{l}.encoders{map};
    sg = net.layers{l}.spikegenerators{map};
    drive = encoders' * states;
    rates = getRates(sg, drive, 0, 0);
end