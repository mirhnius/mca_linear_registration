% set display format to long
format long;

% number of runs per each input and function
n_runs = 100;

% initialize the random seed generator
rand("seed", 42);

% define the list of functions and to test and their  constant inputs

functions_to_test = {
    @sqrt, "sqrt", 2;
    @cbrt, "cbrt", 3;
    @sin, "sin", 1;
    @cos, "cos", 1;
    @asin, "asin", 1;
    @acos, "acos", 0.5;
    @atan, "atan", 1;
    @sinh, "sinh", 1;
    @cosh, "cosh", 1.5;
    @tanh, "tanh", 0.5;
    @asinh, "asinh", 1;
    @acosh, "acosh", 1.5;
    @atanh, "atanh", 0.5;
    @gamma, "gamma", 0.5;
    @lgamma, "lgamma", 0.5;
    @exp, "exp", 0.5;
    @log, "log", 2.1;
    @log10, "log10", 2;
    @log2, "log2", 3;
    @(x) besselj(0,x), "j0", 3;

}
    %@besselj, "j1", 3;
    %@bessely, "y0", 6;
    %@bessely, "y1", 6;
% open the csv file to write the results
fid = fopen("stability_results.csv", "w");
fprintf(fid, "function_name, run, input, result, mean, std\n");

% initialize the results data structure
results = {};

for f_i=1:size(functions_to_test, 1)
    func_handel = functions_to_test{f_i, 1};
    f_name = functions_to_test{f_i, 2};
    f_input = functions_to_test{f_i, 3};

    %  call the function n_runs times and store the output

    func_results = zeros(1, n_runs);
    for j=1:n_runs
        func_results(j) = func_handel(f_input);

        mean_val = mean(func_results(1:j));
        std_val = std(func_results(1:j));

        % write the results in the csv file
        fprintf(fid, "%s, %d, %f, %.18f, %.18f, %.18f\n", f_name, j, f_input, func_results(j), mean_val, std_val);
    end

     % store the results in the results data structure
    results.(f_name) = func_results;

    % final mean and var
    fprintf("Function: %s, Mean: %.18f, Std: %.18f\n", f_name, mean_val, std_val);
end

% close the csv file
fclose(fid);
