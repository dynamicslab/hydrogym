function f = make_fun(arg)
    if isnumeric(arg)
        if isempty(arg)
            f = @(x) x;
        elseif isvector(arg)
            v = reshape(arg,[],1);
            f = @(x) v .* x;
        else
            M = arg;
            f = @(x) M*x;
        end
    else
        f = arg;
    end
end
