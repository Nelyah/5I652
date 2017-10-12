function [bow,nc] = computeBow(sifts, clusters)
    nc = assignementKMeans(sifts, clusters);
    bow = zeros(1001,1);
    nb_sifts = size(nc,1);
    
    
    for i = 1:nb_sifts
        bow(nc(i)) = bow(nc(i)) + 1;
    end
    bow = bow./nb_sifts;

end

