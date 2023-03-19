function [D] = mahal2_full(x)
% D = mahal2(x)
% computes the mahalanobis distance between two distributions in the
% structure x

D=zeros(length(x));
for i=1:length(x)
    A = x{i};
    %A = A(:,65:96);
    m1 = mean(A);
    C1 = cov(A) + 1e-3*eye(size(cov(A)));
    %     if rank(C1) < size(A,2)
    %         [u,s,v] = svd(C1);
    %         r=rank(C1);
    %         iC1 = v(:,1:r) * inv(s(1:r,1:r)) * u(:,1:r)';
    %     else
    %         iC1 = pinv(C1);
    %     end

    for j = i+1:length(x)
        B = x{j};
        %B = B(:,65:96);
        m2 = mean(B);
        C2 = cov(B) + 1e-3*eye(size(cov(B)));
        %         if rank(C2) < size(B,2)
        %             [u,s,v] = svd(C2);
        %             r=rank(C2);
        %             iC2 = v(:,1:r) * inv(s(1:r,1:r)) * u(:,1:r)';
        %         else
        %             iC2 = pinv(C2);
        %         end
        C = (C1+C2)/2;
        %iC = (1/2)*(iC1 - iC1*inv(iC2+iC1)*iC1);
        %D(i,j) = (m1-m2) * iC * (m1-m2)';

        if rank(C) < size(A,2)
            [u,s,v] = svd(C);
            r=rank(C);
            iC = v(:,1:r) * inv(s(1:r,1:r)) * u(:,1:r)';
            D(i,j) = (m1-m2) * iC * (m1-m2)';
        else
            D(i,j) = (m1-m2) * pinv(C) * (m1-m2)';
        end
        D(j,i) = D(i,j);
    end
end


