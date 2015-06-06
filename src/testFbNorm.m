% Testing normalization of log-probability vector: feed back a stabilizing
% nonlinear function to create an attractor near zero. 

d = 5;
p0 = rand(d,1)/2; 

k = 20;

% f = @(x) min(0, -x-.2);
f = @(x) -.4./(1+exp(-10*x)); % sigmoid more stable in neurons

dt = .0005;
time = dt:dt:.05;

logp = zeros(d, length(time));
n = zeros(1, length(time));
tauPSC = .01;
for i = 2:length(time)
    n(i) = (dt/tauPSC) * k * (.2 + sum(f(logp(:,i-1)))); 
    logp(:,i) = (dt/tauPSC)*log(p0) + (1-dt/tauPSC)*logp(:,i-1) + n(i); 
end

figure 
subplot(3,1,1), plot(time, f(logp))
subplot(3,1,2), plot(time, logp), set(gca, 'YLim', [-6 1])
subplot(3,1,3), plot(time, n)

n = 200;
net = Network(dt);
input = FunctionInput(@(t) log(p0));
net.addNode(input);
tauRef = .002;
tauRC = .02;
intercepts = -1 + 2*rand(1,n);
maxRates = 100 + 100*rand(1,n);
tauPSC = .01;
sg = LIFSpikeGenerator(dt, tauRef, tauRC, intercepts, maxRates, 0);    
pop = CosinePopulation(5, sg, 'pop');
pop.addOrigin('x', @(x) x);
pop.addOrigin('fb', @(x) k * (.2/d -.4./(1+exp(-10*x))));
bigpop = CosinePopulation.makeClonedPopulation(pop, d);
inTerm = bigpop.addTermination('in', tauPSC, eye(d));
fbTerm = bigpop.addTermination('fb', tauPSC, ones(d,d));
net.addNode(bigpop);
net.addConnection(input.getOrigin(), inTerm);
net.addConnection(bigpop.getOrigin('fb'), fbTerm);
p = net.addProbe(bigpop.getOrigin('x'), 'output');
net.run(0, .5);
plotProbe(p, tauPSC);
set(gca, 'YLim', [-6 1])
