% This functions subtracts two columns of different sizes such that the column
% with longer length is reduced in size to match the length of the other columns
% input:  x,y = the columsns to subtracted
% output: z = result of subtraction of of x from y

function [z,r]=subtract_columns(x,y)
    
    lx=length(x);
    ly=length(y);
    
    if lx>=ly
        y=[y;zeros(lx-ly,1)];
        z=x-y;
        r=[];
    else
        z=x-y(1:lx);
        r=y(length(x)+1:end);
    end
    
end