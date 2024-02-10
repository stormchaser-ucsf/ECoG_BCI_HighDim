function [A,tau,yhat,tt] = exp_fit(x,y)
%function [A,tau,yhat] = exp_fit(x,y)
% TO FIT AN EXPONENTIAL CURVE TO DATA of model y=A*exp(b*x);

y = log(y);
x = [ones(length(x),1) x(:)];
bhat = x\(y(:));

% returning the parameters of the fit
A = exp(bhat(1));
tau = bhat(2);

tt = linspace(x(1,2),x(end,2),1e4);
yhat = A *exp(tau*tt);






% 
% 
% %%
% 
% % Example data
% x = [1, 2, 3, 4, 5];
% y = [10, 20, 45, 80, 120];
% 
% % Take the natural logarithm of y
% y_log = log(y);
% 
% % Create the design matrix for linear regression
% X = [ones(size(x')), x'];
% 
% % Perform linear regression
% coefficients = X \ y_log';  % Backslash operator for least squares
% 
% % Extract fitted parameters
% ln_A = coefficients(1);
% B = coefficients(2);
% 
% % Calculate A from ln_A
% A = exp(ln_A);
% 
% % Display the fitted parameters
% disp('Fitted Parameters:');
% disp(['A: ', num2str(A)]);
% disp(['B: ', num2str(B)]);
% 
% % Plot the data and the fitted curve
% figure;
% scatter(x, y, 'o', 'DisplayName', 'Data');
% hold on;
% 
% % Plot the fitted curve
% xfit = linspace(min(x), max(x), 100);  % Generating points for smooth curve
% yfit = A * exp(B * xfit);
% plot(xfit, yfit, 'r-', 'DisplayName', 'Exponential Fit');
% 
% xlabel('X');
% ylabel('Y');
% legend('show');
% title('Exponential Curve Fitting (Linear Regression)');
