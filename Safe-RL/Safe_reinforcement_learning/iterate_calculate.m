function X = iterate_calculate( Init, M, N )
% this function iteratively solve for the following equation for X:
% X = M + N'*X*N
% starting from Init

X = Init; diff = 1; iter = 0;
while diff > 1e-3
    iter = iter + 1;
    X_old = X;
    X = M + N'*X*N;
    diff = norm(X_old - X);
end

end

