%{
D-duration unit hydrograph computation

Author: Abhinav Gupta (Created: 30 Dec 2021)
%}

function u_D = uh_D(alpha, beta, D, kmax)
    
    for k = 1:kmax
        u_D(k,1) = 1/D*(gammainc(beta*k*D,alpha) - gammainc(beta*(k-1)*D,alpha));
    end
    
end