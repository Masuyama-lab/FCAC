function noised_X = addLaplacianNoise(X, epsilon)
% Generates random variables from a Laplace distribution
%   epsilon  : privacy budget
%   deltaF   : sensitivity ( abs(v_max - v_min) )
%   default mu = 0
%   scale    : scale parameter = deltaF/epsilon
% Return a dataset X with random variables from a Laplace distribution

delta_F = abs(max(X) - min(X));

if epsilon == 0
    scale = delta_F./1.0e-6;
else
    scale = delta_F./epsilon;
end


mu = 0;


u = rand(size(X)) - 0.5;
lap = mu - scale.*sign(u) .* log(1-2.*abs(u));


noised_X = X + lap;

end