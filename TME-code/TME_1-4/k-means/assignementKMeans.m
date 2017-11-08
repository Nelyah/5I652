function nc = assignementKMeans(X,C)
    [dim_Cx,dim_Cy] = size(C);
    [dim_Xx,dim_Xy] = size(X);
    nb_clusters = dim_Cx;

    NX = X.^2;
    NX = sum(NX,2);
    NX = repmat(NX, 1, dim_Cx);

    NC = C.^2;
    NC = sum(NC,2);
    NC = repmat(NC, 1, dim_Xx);
    NC = NC';

    dist_mat = NX + NC - (2 * mtimes(X,C'));

    [~, nc] = min(dist_mat, [], 2);
end