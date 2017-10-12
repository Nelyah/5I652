%% init aleatoire de données jouet

clear all;
close all;
clc;
nb_center = 1000;
min_err = -1;
nb_run = 1;

points = rand(10000, 2);
[points, norms] = randomSampling('../descriptors_files/');

for i = 1:nb_run
    [C, ERR] = solutionKMeans(points, nb_center);
    tmp_err = sum(ERR)/nb_center;
    if min_err == -1
       min_err = tmp_err
       solution_C = C;
       continue
    end
    if tmp_err < min_err
        min_err = tmp_err
        solution_C = C;
    end
end

empty_norm = zeros(1,128);
solution_C = [solution_C ; empty_norm];

file_out = fopen('visual_dictionary.txt', 'wt')
for i = 1:size(solution_C, 1)
    fprintf(file_out, '%g\t', solution_C(i, :));
    fprintf(file_out, '\n');
end
fclose(file_out);
%save('visual_dictionary.txt','solution_C');

% plot(points(:,1), points(:,2), '.')
% hold on
% plot(C(:,1), C(:,2), 'o')