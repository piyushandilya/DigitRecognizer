function imclean = imclean(im)    vecIm = im(:);    imclean = zeros(size(vecIm));    for i = 1: size(vecIm)    if(vecIm(i) > -0.5)      imclean(i) = vecIm(i);      else      imclean(i) = 0;      endif  endfor       imclean = reshape(imclean, 25, 25);  
endfunction
