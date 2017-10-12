function [newcenters , erreur , movecenters ] = miseAjourKMeans( points, centers , nc)
    [nb_centers, dim] = size(centers);
    newcenters = zeros(nb_centers, dim);
    erreur = zeros(nb_centers, 1);
    movecenters = zeros(nb_centers, dim);
    
    % For each center
    for idx_center = 1:nb_centers
       c = points(nc==idx_center,:);
       [c_size, ~] = size(c);
       center = mean(c, 1);
       newcenters(idx_center,:) = center;
       n = sum(sqrt(sum((c - repmat(center, c_size, 1)).^2, 2)));
       erreur(idx_center) = n;
       movecenters(idx_center,:) = center - centers(idx_center,:);
    end
end