%
% pmfg calculate the the planar maximally filtered  graph (PMFG) 
% from a matrix of weights W. 
% P = pmfg(W) returns a sparse matrix with P(i,j)=W(i,j) if the edge i-j 
% is present and P(i,j)=0 if not.
% W should be sparse, real, square and symmetric matrix.
%
% Copyright 2010 Tomaso Aste
% The University of Kent, UK
% t.aste(at)kent.ac.uk
%
% This function uses "matlab_bgl" package from 
% http://www.stanford.edu/~Edgleich/programs/matlab_bgl/
%
% Please reference
%
% T. Aste, T. Di Matteo and S. T. Hyde, 
% "Complex Networks on Hyperbolic Surfaces", 
% Physica A 346 (2005) 20-26. 
%
% M. Tumminello, T. Aste, T. Di Matteo, R.N. Mantegna, 
% "A tool for filtering information in complex systems", 
% Proceedings of the National Academy of Sciences of the United States 
% of America (PNAS) 102 (2005) 10421-10426. 
%
% in your published research.

%  
%
%-----------------------------------------------------------------------------------------
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should receive a copy of the GNU General Public License
% along with this program.  See also <http://www.gnu.org/licenses/>.
%-----------------------------------------------------------------------------------------
 
 
function P = pmfg(W)
 
if size(W,1)~=size(W,2)
    fprintf('W must be square \n');
    P =[];
    return
end
if ~isreal(W)
    fprintf('W must be real \n');
    P =[];
    return
end
if any(any(W-W'))
    fprintf('W must be symmetric \n');
    P =[];
    return
end
if ~issparse(W)
    W = sparse(W);
end
N = size(W,1);
if N == 1
    P = sparse(1);
    return
end
[i,j,w] = find(sparse(W));
kk = find(i < j);
ijw= [i(kk),j(kk),w(kk)];
ijw = -sortrows(-ijw,3); %make a sorted list of edges
P = sparse(N,N);
for ii =1:min(6,size(ijw,1)) % the first 6 edges from the list can be all inserted in a tetrahedron
    P(ijw(ii,1),ijw(ii,2)) = ijw(ii,3);
    P(ijw(ii,2),ijw(ii,1)) = ijw(ii,3);
end
E = 6; % number of edges in P at this stage
P1 = P;
while( E < 3*(N-2) ) % continue while all edges for a maximal planar graph are inserted
    ii = ii+1;
    P1(ijw(ii,1),ijw(ii,2))=ijw(ii,3); % try to insert the next edge from the sorted list
    P1(ijw(ii,2),ijw(ii,1))=ijw(ii,3); % insert its reciprocal
    if boyer_myrvold_planarity_test(P1~=0) % is the resulting graph planar?
        P = P1; % Yes: insert the edge in P
        E = E+1;
    else
        P1 = P; % No: discard the edge
    end
    if floor(ii/1000)==ii/1000; 
        %save('P.mat','P','ii')
        fprintf('Build P: %d    :   %2.2f per-cent done\n',ii,E/(3*(N-2))*100);
        if ii > (N*(N-1)/2)
            fprintf('PMFG not found \n');
            P = full(P);
            return
        end
    end
end
P = full(P)
end
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% to plot it you can use %%%%%%%
% xy = chrobak_payne_straight_line_drawing(P);
% figure
% gplotwl(P,xy,Labels,'-b');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

