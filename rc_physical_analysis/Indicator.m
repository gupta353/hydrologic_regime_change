% Indicator function
% inputs: x = argument
%         ll = lower limit
%         ul = upper limit
% outputs: I = Indicator function value

function I = Indicator(x,ll,ul)

    I = x>=ll & x<=ul;
    I = double(I);
    
end