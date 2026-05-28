function el = meanEdgeLength(V, F)
%MEANEDGELENGTH  Mean triangle edge length of a mesh (for arrow scaling).
    E = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])];
    el = mean(vecnorm(V(E(:,1),:) - V(E(:,2),:), 2, 2));
end
