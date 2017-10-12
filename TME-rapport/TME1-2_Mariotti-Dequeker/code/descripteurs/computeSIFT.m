function sift=computeSIFT(s,Ig,Ior,Mg)
    sift = [];
    for x = 0:3
        for y = 0:3
            hist = zeros(8,1);
            hist = hist(:);
            for x1 = 1:4
                for y1 = 1:4
                    cx = 4*x + x1;
                    cy = 4*y + y1;
                    if Ior(cx, cy) ~= 0
                        hist(Ior(cx,cy),1) = hist(Ior(cx,cy),1) + Ig(cx,cy)*Mg(cx,cy);
                    end
                end
            end
            hist_norm = norm(hist);
            
            if hist_norm > 0.2
                hist = hist/hist_norm;
                hist = max(hist, 0.2);
                hist_norm = sqrt(sum(hist.^2, 1));
                hist = hist/hist_norm;
            else
                hist = zeros(8,1);
            end
            sift = [sift; hist];
        end
    end
end
