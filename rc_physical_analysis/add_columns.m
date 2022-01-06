% This functions adds two columns of different sizes such that the column
% with shorter length is appended below with zeros to match the length of
% the other columns
% input:  x,y = the columsns to be added
% output: z = result of addition of x and y
function z=add_columns(x,y)
    
    lx=length(x);
    ly=length(y);
    
    if (lx==0) 
        x=[]; 
    end
    
    if (ly==0) 
        y=[]; 
    end
        
    if lx>ly
        diff_length=lx-ly;
        y=[y;zeros(diff_length,1)];
    elseif lx<ly
        diff_length=ly-lx;
        x=[x;zeros(diff_length,1)];        
    end
    z=x+y;
end