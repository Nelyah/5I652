function [centres,erreur] = solutionKMeans(points,K)
    centres = randomSeed(points,K);
    erreur = centres;
    movecenters = centres;
    
    while any(movecenters(:))
        % Assign each point to its closest center
        nc = assignementKMeans(points, centres);
        [centres , erreur , movecenters ] = miseAjourKMeans( points, centres , nc);
    end
end