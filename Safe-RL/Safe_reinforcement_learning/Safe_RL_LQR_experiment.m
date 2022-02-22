%% Codes for the constrained Linear-Quadratic Regulator (LQR) experiment of the following paper:
%  Convergent Policy Optimization for Safe Reinforcement Learning, in NeurIPS 2019
%  Authors: Ming Yu, Zhuoran Yang, Mladen Kolar, and Zhaoran Wang 

% We focus on LQR with random initial state, as discussed in Section 5 in the paper 
% x is state, u is control with u = - F * x
% F is the parameter matrix we would like to optimize
% x_{t+1} = A * x_t + B * u_t 
% u_t = - F * x_t
% x_0 ~ D
% D is the unit cone: [-1, 1]^nx
% A: nx * nx    B: nx * nu     F: nu * nx
% Q: nx * nx    R: nu * nu

%% Generate parameters

% dimension of n and u
nx = 15; nu = 8;
A = randn(nx,nx); A = (A+A')/30;
B = randn(nx, nu) / 3; 

% generate matrices: (Q1,R1) should be very different from (Q2,R2)
C1 = rand(nx) .* rand(nx) + 0.5; E1 = randn(nu);
Q1 = gallery('randcorr',nx) * 6; R1 = E1*E1';
C2 = exprnd(1/3,nx,nx); E2 = rand(nu) .* (rand(nu)*2 + 0.3);
Q2 = C2*C2'; R2 = gallery('randcorr',nu); R2 = R2 * R2';

% solve for the unconstrained solution as J0, find the upper bound for D0 as D0_upper
% if D0 >= D0_upper, then it is as if no constraint
[P,L,K] = dare(A,B,Q1,R1);
PK = iterate_calculate(Q2, Q2 + K'*R2*K, (A-B*K));
D0_upper = trace(PK)/3; % this is the upper bound. 
PK_J = iterate_calculate(Q1, Q1 + K'*R1*K, (A-B*K));
J0 = trace(PK_J)/3; % unconstrained minimum

% solve for D as objective, find lower bound for D0 as D0_lower
% if D0 < D0_upper, then it is infeasible
[P,L,KD] = dare(A,B,Q2,R2);
PK = iterate_calculate(Q2, Q2 + KD'*R2*KD, (A-B*KD));
D0_lower = trace(PK)/3; % this is the lower bound. 

% find the constraint value when F = 0
PF_0 = iterate_calculate(Q2, Q2, A);
D0_0 = trace(PF_0)/3; % this is the initialization when F = 0

% choose a reasonable D0
D0 = (D0_upper + D0_lower) / 2;
if D0_0 <= D0
    D0 = (D0_0 + D0_upper) / 2;
end


%% begin algorithm

bar_J_const = 0; bar_J_linear = 0; bar_J_quadratic = 0;
bar_D_const = 0; bar_D_linear = 0; bar_D_quadratic = 0;

t = 0; tau = 5; F = zeros(nu, nx); % initialize F as all-zero matrix
D_all = []; J_all = []; % record constraint and objective value in each iteration
F_all = []; Feasible = []; Feasible_D = []; 
while 1
    
    t = t + 1;
    % max number of iterations
    if t > 3000
        break
    end
    % step size
    rho_t = t ^ (-2/3) / 1.5; eta_t = t ^ (-3/4) / 1.5;

    % for negative reward funtion
    PF = iterate_calculate(Q1, Q1 + F'*R1*F, (A-B*F));
    
    % for cost function
    PF_D = iterate_calculate(Q2, Q2 + F'*R2*F, (A-B*F));

    % sample x0_star ~ D and calculate; we sample 20 replicates and take the average
    IN = 20;
    J_star = 0; D_star = 0; grad_J_star = zeros(nu,nx); grad_D_star = zeros(nu,nx); 
    for in = 1:IN
        x0_star = rand(nx,1) * 2 - 1;
        SF = iterate_calculate(x0_star * x0_star', x0_star * x0_star', (A-B*F));

        J_star = J_star + x0_star' * PF * x0_star / IN;
        grad_J_star = grad_J_star + 2 * ( (R1 + B'*PF*B) * F - B'*PF*A ) * SF / IN;

        D_star = D_star + x0_star' * PF_D * x0_star / IN;
        grad_D_star = grad_D_star + 2 * ( (R2 + B'*PF_D*B) * F - B'*PF_D*A ) * SF / IN;
    end
    
    % For this problem, we can explicitly calculate the objective and constraint values
    % calculate constraint value for this F
    D_all(t) = trace(PF_D)/3;
    Feasible_D(t) = trace(PF_D)/3 <= D0;
    
    % calculate objective value for this F
    J_all(t) = trace(PF)/3;
    
    
    % solve for QCQP
    % objective
    tilde_J_const = J_star - trace(grad_J_star' * F) + tau * trace(F' * F);
    tilde_J_linear = grad_J_star - 2 * tau * F;
    tilde_J_quadratic = tau;

    bar_J_const = (1-rho_t) * bar_J_const + rho_t * tilde_J_const;
    bar_J_linear = (1-rho_t) * bar_J_linear + rho_t * tilde_J_linear;
    bar_J_quadratic = (1-rho_t) * bar_J_quadratic + rho_t * tilde_J_quadratic;

    % constraint
    tilde_D_const = D_star - trace(grad_D_star' * F) + tau * trace(F' * F);
    tilde_D_linear = grad_D_star - 2 * tau * F;
    tilde_D_quadratic = tau;

    bar_D_const = (1-rho_t) * bar_D_const + rho_t * tilde_D_const;
    bar_D_linear = (1-rho_t) * bar_D_linear + rho_t * tilde_D_linear;
    bar_D_quadratic = (1-rho_t) * bar_D_quadratic + rho_t * tilde_D_quadratic;

    % check if feasible
    % straightforward to check since quadratic term is Identity matrix
    
    lower_bound = bar_D_const - sumsqr(bar_D_linear)/4/bar_D_quadratic;
    
    % if feasible, solve for QCQP
    if lower_bound <= D0
        
        % QCQP dimension is nu * nx
        Q = 2 * tilde_J_quadratic * eye(nu * nx);
        f = reshape(tilde_J_linear, nu * nx, 1);
        c = bar_J_const;

        H{1} = 2 * tilde_D_quadratic * eye(nu * nx);
        k{1} = reshape(tilde_D_linear, nu * nx, 1);
        d{1} = bar_D_const - D0;

        options = optimoptions(@fmincon,'Display','off','Algorithm','interior-point',...
            'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
            'HessianFcn',@(x,lambda)quadhess(x,lambda,Q,H));
        options.OptimalityTolerance = 3e-4;
        options.ConstraintTolerance = 3e-4;

        objfun = @(x)quadobj(x,Q,f,c);
        nonlconstr = @(x)quadconstr(x,H,k,d);
        x0 = reshape(F, nu * nx, 1); % column vector
        [x,fval,eflag,output,lambda] = fmincon(objfun,x0,[],[],[],[],[],[],nonlconstr,options);
        
        F_bar = reshape(x, nu, nx);
        Feasible(t) = 1;
        
    % otherwise, solve for feasibility problem
    else
        x = - bar_D_linear/2/bar_D_quadratic;
        F_bar = reshape(x, nu, nx);
        Feasible(t) = 0;
    end
    
    % update parameter matrix F
    F = (1-eta_t) * F + eta_t * F_bar;
    % record F
    F_all(:,t) = reshape(F, nu * nx, 1);
    
    if mod(t,200) == 0
        fprintf(' %d ',t/200);
    end
    
end


%% postprocess, find the minimum
feasible = D_all <= D0;
J_all_inf = J_all; J_all_inf(~feasible) = inf;
obj = min( J_all(feasible) ); % the minimum objective values when the parameter is feasible
place = find( J_all_inf == min( J_all(feasible) ) );
% approximate version
place_almost = find( J_all_inf <= 1.0002 * min( J_all(feasible) ) , 1 ); 
obj_almost = J_all_inf(place_almost);


%% plot the objective and constraint values in each iterate

figure;
subplot(1,2,1)
plot(1:t-1, D_all,'LineWidth',2.4)
hold on;
line([1 t-1],[D0 D0],'LineWidth',2,'color','red','LineStyle', '--')
xlabel('iteration');ylabel('constraint value');
set(gca,'FontSize',15);
set(get(gca,'YLabel'),'Fontsize',25) 
set(get(gca,'XLabel'),'Fontsize',25)

subplot(1,2,2)
plot(1:t-1, J_all,'LineWidth',2.4)
hold on;
line([1 t-1],[J0 J0],'LineWidth',2,'color','red','LineStyle', '--') % unconstrained minimum
xlabel('iteration');ylabel('objective value');
set(gca,'FontSize',15);
set(get(gca,'YLabel'),'Fontsize',25) 
set(get(gca,'XLabel'),'Fontsize',25)


