function DEMO_GenerativeModelsActiveVision
% This demo is associated with the paper 'Generative models for active
% vision' (Parr, Sajid, Da Costa, Mirza, and Friston). It sets out the
% series of steps that could be used by the brain's implicit generative
% model to predict a retinal image. The idea is that each of these
% processes may be seen as factors in a probabilistic generative model that
% could in principle be inverted to recover the message passing in the
% visual system.

% Initialisation
%--------------------------------------------------------------------------
rng default  % For reproducibility

% Priors
%--------------------------------------------------------------------------
% These are categorical distributions over some of the key variables in the
% model. Here, we factorise priors over eye direction into two spatial
% scales, similar to the approach used in Parr and Friston, 2018
% (PMID 29190328) to model visual neglect.

D1 = [0 1 0]';           % Priors over rooms
D2 = [0 1 0 0 0 0 0 0]'; % Priors over heading direction
D3 = [0 1 0 0]';         % Priors over eye direction (course scale)
D4 = [0 0 1 0]';         % Priors over eye direction (fine scale)
D5 = [1 0 0 0 0]';       % Prior over head location

% THE WHAT PATHWAY
%==========================================================================
% Alternative rooms
%--------------------------------------------------------------------------
% Rooms defined as having 2 objects (each of which is chosen from 2
% possible objects) in 2 locations (out of four possibilities).

R{1} = [0 0 2 1]; % Room 1
R{2} = [0 2 0 1]; % Room 2
R{3} = [0 2 2 0]; % Room 3

% Likelihood (of object being in view)
%--------------------------------------------------------------------------
% This is not strictly necessary for what follows, but eliminates surfaces
% that a priori are not in view, based upon the discrete variables above.
% This confers a slight improvement in the speed with which the model may
% be evaluated.

for f1 = 1:length(D1)
    for f2 = 1:length(D2)
        for f3 = 1:length(D3)
            for f4 = 1:length(D4)
                for f5 = 1:length(D5)
                    A1(1,f1,f2,f3,f4,f5) = 1;
                    A2(1,f1,f2,f3,f4,f5) = 1;
                    if find(R{f1},1) == 1
                        if f2 == 1 || f2 == 2 || f2 == 3
                            A1(2,f1,f2,f3,f4,f5) = 1;
                        end
                    elseif find(R{f1},1) == 2
                        if f2 == 1 || f2 == 8 || f2 == 9
                            A1(2,f1,f2,f3,f4,f5) = 1;
                        end
                    elseif find(R{f1},1) == 3
                        if f2 == 5 || f2 == 6 || f2 == 7
                            A1(2,f1,f2,f3,f4,f5) = 1;
                        end
                    end
                    if find(R{f1},1,'last') == 2
                        if f2 == 1 || f2 == 8 || f2 == 9
                            A2(2,f1,f2,f3,f4,f5) = 1;
                        end
                    elseif find(R{f1},1,'last') == 3
                        if f2 == 3 || f2 == 4 || f2 == 5
                            A2(2,f1,f2,f3,f4,f5) = 1;
                        end
                    elseif find(R{f1},1,'last') == 4
                        if f2 == 5 || f2 == 6 || f2 == 7
                            A2(2,f1,f2,f3,f4,f5) = 1;
                        end
                    end    
                end
            end
        end
    end
end

% Sample objects and locations from room
%--------------------------------------------------------------------------
% This generates a matrix OL whose first row indexes identities, and whose
% second row indexes locations.

r  = find(rand < cumsum(D1),1); % Sample from prior
OL = [R{r}(R{r}>0);find(R{r})]; % Specify objects and their locations

A1 = A1(:,r,:,:,:,:);
A2 = A2(:,r,:,:,:,:);

% Define the 2 objects in terms of their constituent parts (assumed
% spherical)
%--------------------------------------------------------------------------
% Here we collate the surfaces of objects defined by sampling from the
% above priors. Each object is defined in terms of a collection of spheres
% that each undergo affine transformations. The collection of spheres
% comprising an object then undergo similar transformations, placing them
% in the room.

T   = [ 13, 13,0;   % Possible translations (rows)
        13,-13,0;
       -13, 13,0;
       -13,-13,0];
XYZ = {};           % Surfaces in scene

% Loop through objects
%--------------------------------------------------------------------------
% Here we loop through the objects identified above, and find their
% constituent surfaces in mesh form.

for i = 1:size(OL,2)
    xyz = gmav_object(OL(1,i));   % Get surfaces of objects
    Q = randn(3,1)/16;            % Rotation
    S = randn(3,1)/16 + 3;        % Scaling
    for j = 1:size(xyz,1)
        [X,Y,Z] = gmav_rotate_scale_translate(Q,S,T(OL(2,i),:)',xyz{j,1},xyz{j,2},xyz{j,3});
        XYZ = [XYZ; {X,Y,Z}];
    end
end

% Add Ground
%--------------------------------------------------------------------------
[Xg, Yg] = meshgrid(-25:2.5:25,-25:2.5:25);
Xg = [Xg Xg(:,end)];
Yg = [Yg Yg(:,end)];
Zg       = -5*ones(size(Xg));

XYZ = [XYZ;{Xg,Yg,Zg}];

% Plot mesh of complete scene
%--------------------------------------------------------------------------
% This displays the surfaces of the scene. We can see everything we have
% done so far as part of the ventral visual pathway, concerned with what is
% in our environment. The next step will be to establish the position of
% the retina in space, which calls upon the dorsal visual pathway, and
% those parts of the brain involved in spatial cognition.

figure('Name','Mesh','Color','w'), clf
for k = 1:size(XYZ,1)
    mesh(XYZ{k,1},XYZ{k,2},XYZ{k,3},'EdgeColor','k'), axis equal, hold on
end

% THE WHERE PATHWAY
%==========================================================================
% Retinal location
%--------------------------------------------------------------------------
% This depends upon 3 things: the location in space (allocentric), the
% direction the head is pointing (allocentric), and the angle of the eyes
% in the head (egocentric)

% 1. Sample head location
%--------------------------------------------------------------------------
r  = find(rand < cumsum(D5),1); % Sample from prior
L = [ 0, 0,1/2;                 % Head locations
      5, 5,1/2;
      5,-5,1/2;
     -5, 5,1/2;
     -5,-5,1/2];
L = L(r,:);

A1 = A1(:,:,:,:,:,r);
A2 = A2(:,:,:,:,:,r);

% 2. Sample head direction
%--------------------------------------------------------------------------
r  = find(rand < cumsum(D2),1); % Sample from prior
H = [(r-1)*pi/4,0];             % Allocentric heading direction

A1 = A1(:,:,r,:,:);
A2 = A2(:,:,r,:,:);

% 3. Sample eye direction
%--------------------------------------------------------------------------
% This involves sampling from both the coarse and fine scale, and
% accounting for convergence of the eyes. 

% 3a. Sample from coarse scale
%--------------------------------------------------------------------------
r  = find(rand < cumsum(D3),1); % Sample from prior
Ec = [];
for i = 1:2
    for j = 1:2
        Ec(end+1,:) = [-pi/16 + (i-1)*pi/8,-pi/16 + (j-1)*pi/8];
    end
end
Ec = Ec(r,:);
A1 = A1(:,:,:,r,:);
A2 = A2(:,:,:,r,:);

E  = [];
for i = 1:2
    for j = 1:2
        E(end+1,:) = Ec + [-pi/32 + (i-1)*pi/16,-pi/16 + (j-1)*pi/32];
    end
end

% 3b. Sample from fine scale
%--------------------------------------------------------------------------
r  = find(rand < cumsum(D4),1);             % Sample from prior
E  = [E(r,:)+[-0.01,0];E(r,:)+[+0.01,0]];   % Adjust for convergence

A1 = A1(:,:,:,:,r);
A2 = A2(:,:,:,:,r);

% Construct retina
%--------------------------------------------------------------------------
% With the above information available, we can now construct a retinal
% array--i.e., an array of cells (or groups of cells) associated with light
% hitting the lens at different angles.

% Constants
%--------------------------------------------------------------------------
f = pi/128;     % foveal field of view
ds = 1.5;       % Spacing between eyes

% Construct and plot retina
%--------------------------------------------------------------------------
% Here, we construct our array of cells, projected out through the lens. We
% then add these to the figure, placing the retina and the surfaces in the
% same space. In addition, we plot the eyes, and a circle representing the
% head position, for ease of visualisation.

V  = gmav_retinal_array(L,H,E,f/2,65,ds); 
plot3(V.L(:,:,1),V.L(:,:,2),V.L(:,:,3),'.r'), hold on
plot3(V.R(:,:,1),V.R(:,:,2),V.R(:,:,3),'.b')
gmav_plot_eyes([E(1,:),0],[E(2,:),0],H(1),L,ds);
axis equal

% THE RETINOCORTICAL PATHWAY
%==========================================================================
% This section computes the image expected on the retina, given the retinal
% location and surfaces sampled above.

% Visual field defects
%==========================================================================
% This section of code (commented out) omits all surfaces on one side of
% egocentric space, illustrating the consequences of a disconnection
% between the visual cortex (representing surfaces) and the retina, while
% still in the space of surfaces (i.e., on the visual cortical side of the
% decussation at the chiasm). This results in a homonymous hemianopia.
%
%--------------------------------------------------------------------------
% for i = 1:size(XYZ,1)
%     for j = 1:size(XYZ{i,1},1)
%         for k = 1:size(XYZ{i,1},2)
%             v = [XYZ{i,1}(j,k),XYZ{i,2}(j,k),XYZ{i,3}(j,k)] - L;
%             v = v/sqrt(v*v');
%             if v(1:2)*[cos(mean(E(:,1))+H(1)+pi/2);sin(mean(E(:,1))+H(1)+pi/2)] < 0, K(i,j,k) = 1; end
%         end
%     end
% end
%==========================================================================

tic

% Light source direction
%--------------------------------------------------------------------------
LS = [3*pi/2,-pi/2];

% Iterate over surfaces (starting from each retinal cell)
%--------------------------------------------------------------------------
Lr  = 32;                                 % length of vectors shown from retinal array (for plotting)
I.L = zeros(size(V.L,1),size(V.L,2));     % activation of cells in left retina
I.R = zeros(size(V.R,1),size(V.R,2));     % activation of cells in left retina

for c1 = 1:size(V.L,1)      % iterate along columns of cells in retina
    for c2 = 1:size(V.L,2)  % iterate along rows of cells in retina
        
        % Left retina
        %------------------------------------------------------------------
        x = [V.L(c1,c2,1),V.L(c1,c2,2),V.L(c1,c2,3)];
        n = x - [L(1) + ds*cos(H(1)+pi/2), L(2) + ds*sin(H(1)+pi/2), L(3)]; % Retinal cell location minus lens location
        n = n/sqrt(n*n');
        if c1 == 1 || c1 == size(V.L,1)
            if c2 == 1 || c2 == size(V.L,2)
                plot3([x(1),x(1) + Lr*n(1)],[x(2),x(2) + Lr*n(2)],[x(3),x(3) + Lr*n(3)],'r'), hold on
            end
        end
        if A1(2) > 0 && A2(2) > 0
            r = gmav_find_surface(x,n,XYZ);
            try, if K(r(1),r(2),r(3)), r = [];end, end
            I.L(c1,c2) = gmav_apply_lighting(r,n,XYZ,LS,0.5,1,2);
        elseif A1(2) > 0
            try, J = cat(1,K(1:3,:,:),K(end,:,:));, end
            r = gmav_find_surface(x,n,[XYZ(1:3,:);XYZ(end,:)]);
            try, if J(r(1),r(2),r(3)), r = [];end, end
            I.L(c1,c2) = gmav_apply_lighting(r,n,[XYZ(1:3,:);XYZ(end,:)],LS,0.5,1,2);
        elseif A2(2) > 0
            try, J = cat(1,K(4:6,:,:),K(end,:,:)); end
            r = gmav_find_surface(x,n,[XYZ(4:6,:);XYZ(end,:)]);
            try, if J(r(1),r(2),r(3)), r = [];end, end
            I.L(c1,c2) = gmav_apply_lighting(r,n,[XYZ(4:6,:);XYZ(end,:)],LS,0.5,1,2);
        else
            try, J = K(end,:,:); end
            r = gmav_find_surface(x,n,XYZ(end,:));
            try, if J(r(1),r(2),r(3)), r = [];end, end
            I.L(c1,c2) = gmav_apply_lighting(r,n,XYZ(end,:),LS,0.5,1,2);
        end
        
        % Right retina
        %------------------------------------------------------------------
        x = [V.R(c1,c2,1),V.R(c1,c2,2),V.R(c1,c2,3)];
        n = x - [L(1) - ds*cos(H(1)+pi/2), L(2) - ds*sin(H(1)+pi/2), L(3)]; % Retinal cell location minus lens location
        n = n/sqrt(n*n');
        if c1 == 1 || c1 == size(V.L,1)
            if c2 == 1 || c2 == size(V.L,2)
                plot3([x(1),x(1) + Lr*n(1)],[x(2),x(2) + Lr*n(2)],[x(3),x(3) + Lr*n(3)],'b'), hold on
            end
        end
        if A1(2) > 0 && A2(2) > 0
            r = gmav_find_surface(x,n,XYZ);
            try, if K(r(1),r(2),r(3)), r = [];end, end
            I.R(c1,c2) = gmav_apply_lighting(r,n,XYZ,LS,0.5,1,2);
        elseif A1(2) > 0
            try, J = cat(1,K(1:3,:,:),K(end,:,:)); end
            r = gmav_find_surface(x,n,[XYZ(1:3,:);XYZ(end,:)]);
            try, if J(r(1),r(2),r(3)), r = [];end, end
            I.R(c1,c2) = gmav_apply_lighting(r,n,[XYZ(1:3,:);XYZ(end,:)],LS,0.5,1,2);
        elseif A2(2) > 0
            try, J = cat(1,K(4:6,:,:),K(end,:,:)); end
            r = gmav_find_surface(x,n,[XYZ(4:6,:);XYZ(end,:)]);
            try, if J(r(1),r(2),r(3)), r = [];end, end
            I.R(c1,c2) = gmav_apply_lighting(r,n,[XYZ(4:6,:);XYZ(end,:)],LS,0.5,1,2);
        else
            try, J = K(end,:,:); end
            r = gmav_find_surface(x,n,XYZ(end,:));
            try, if J(r(1),r(2),r(3)), r = [];end, end
            I.R(c1,c2) = gmav_apply_lighting(r,n,XYZ(end,:),LS,0.5,1,2);
        end
    end
end

% Discrete cosine transforms
%--------------------------------------------------------------------------
DCT = dctmtx(size(I.R,1));
DR  = DCT*I.R*DCT';
DL  = DCT*I.L*DCT';

% Attenuate high frequency components
%--------------------------------------------------------------------------
[U1, U2] = meshgrid(1:size(DR,1),1:size(DR,2));    % Array coordinates
U3       = exp( - ((U1-1).^2 + (U2-1).^2)/256);    % Gaussian centred on low frequencies

DR = DR.*U3;
DL = DL.*U3;

% Inverse discrete cosine transform
%--------------------------------------------------------------------------
I.L = DCT'*DL*DCT;
I.R = DCT'*DR*DCT;

toc

% Plot retinal images
%--------------------------------------------------------------------------
% This shows the result of the generative process above, presenting the
% retinal images (inverted across the horizontal and vertical axes)
% anticipated under a given combination of the variables outlined above.

figure('Name','Retinal array','Color','w'), clf
subplot(1,2,1)
imagesc(I.L), axis square, title('Left retina')
colormap gray, caxis([0 max(max(max(I.L)),max(max(I.R)))]);
subplot(1,2,2)
imagesc(I.R), axis square, title('Right retina')
colormap gray, caxis([0 max(max(max(I.L)),max(max(I.R)))]);

% EYE MOVEMENTS AND THE OCULOMOTOR BRAINSTEM
%==========================================================================
% Finally, we briefly illustrate the generation of proprioceptive data we
% would anticipate on moving from one fixation location to the next. This
% represents the model employed by the oculomotor brainstem.

% Change priors to reflect new anticipated fixation location
%--------------------------------------------------------------------------
D3 = [0 0 1 0];
D4 = [1 0 0 0];

% Save previous fixation location, and sample new location
%--------------------------------------------------------------------------
E1 = mean(E);
r  = find(rand < cumsum(D3),1); % Sample from prior
Ec = [];
for i = 1:2
    for j = 1:2
        Ec(end+1,:) = [-pi/16 + (i-1)*pi/8,-pi/16 + (j-1)*pi/8];
    end
end
Ec = Ec(r,:);
E2  = [];
for i = 1:2
    for j = 1:2
        E2(end+1,:) = Ec + [-pi/32 + (i-1)*pi/16,-pi/16 + (j-1)*pi/32];
    end
end
r  = find(rand < cumsum(D4),1); % Sample from prior
E2  = E2(r,:);

% Equations of motion
%--------------------------------------------------------------------------
% These equations are based upon Newton's second law applied to rotational
% forces.

% First temporal derivative (ode)
%--------------------------------------------------------------------------
f = @(x) [x(3)...
          x(4)...
          E2(1) - x(1) - 2*x(3)...
          E2(2) - x(2) - 2*x(4)]';

% Jacobian of f(x)
%--------------------------------------------------------------------------
J = [0 0  1  0;
     0 0  0  1;
     1 0 -2  0;
     0 1  0 -2];

% Initialise at previous fixation and set integration timestep
%--------------------------------------------------------------------------
x  = [E1 0 0]';
dt = exp(-2);

% Numerically integrate (using matrix exponentials as in (Ozaki 1992))
%--------------------------------------------------------------------------
for t = 1:64
    dx = J\(expm(J*dt)-eye(4))*f(x(:,end));
    x(:,end+1)  = x(:,end) + dx + randn(4,1)/512; % Add small amplitude fluctuations
end

% Plot timecourse of hidden states (oculomotor angles and velocities)
%--------------------------------------------------------------------------
figure('Name','Occulomotion (hidden states)','Color','w'), clf
subplot(1,2,1), subplot(2,1,1), plot(x(1:2,:)'),subplot(2,1,2), plot(x(3:4,:)')

% Define expected value of proprioceptive data, as function of hidden
% states
%--------------------------------------------------------------------------
g = @(x) [x(1,:) - 0.01;
          x(2,:);
          x(1,:) + 0.01;
          x(2,:);
          x(3,:);
          x(4,:);
          x(3,:);
          x(4,:)];
      
% Plot anticipated proprioceptive data
%--------------------------------------------------------------------------
figure('Name','Occulomotion (hidden states)','Color','w'), clf
subplot(2,1,1), plot(g(x)' + randn(size(g(x)'))/512)

end

function [x,y,z] = gmav_rotate_scale_translate(Q,S,T,X,Y,Z)
% This routine applies the rotation (Q), scaling (S), and translation (T)
% transforms to the surfaces given by the X, Y, and Z coordinates.
%--------------------------------------------------------------------------

% Initialise
%--------------------------------------------------------------------------
Qx       = Q(1);
Qy       = Q(2);
Qz       = Q(3);
S        = diag(S);
A(:,:,1) = X;
A(:,:,2) = Y;
A(:,:,3) = Z;
A        = permute(A,[3,1,2]);

% Rotation matrices
%--------------------------------------------------------------------------
Rx = [1 0 0; 0 cos(Qx) -sin(Qx); 0 sin(Qx) cos(Qx)];
Ry = [cos(Qy) 0 sin(Qy); 0 1 0; -sin(Qy) 0 cos(Qy)];
Rz = [cos(Qz) -sin(Qz) 0; sin(Qz) cos(Qz) 0; 0 0 1];

R = Rx*Ry*Rz;

% Iterate through coordinates and apply transforms
%--------------------------------------------------------------------------
for i = 1:size(A,2)
    for j = 1:size(A,3)
        A(:,i,j) = R*S*A(:,i,j) + T;
    end
end

% Return in x,y,z format
%--------------------------------------------------------------------------
A = permute(A,[2,3,1]);
x = A(:,:,1);
y = A(:,:,2);
z = A(:,:,3);

end

function V = gmav_retinal_array(L,H,E,f,N,ds)
% Generate an array for each retina, based upon head location (L), head
% direction (H), eye direction (E), field of view (f), the desired number
% of cells in a row and column (N), and the spacing between the eyes (ds).
%--------------------------------------------------------------------------

% Location of each lens given by head location plus ds in directions
% orthogonal to head direction
%--------------------------------------------------------------------------
LL =    ds*[cos(H(1)+pi/2),sin(H(1)+pi/2),0]; % left lens
LR =  - ds*[cos(H(1)+pi/2),sin(H(1)+pi/2),0]; % right lens

% Centre of virtual retina (accounting for lens) is projection on to angle
% of line of sight
%--------------------------------------------------------------------------
V0 = [cos(E(:,1) + H(1)).*cos(E(:,2) + H(2)),...
      sin(E(:,1) + H(1)).*cos(E(:,2) + H(2)),...
      sin(E(:,1) + H(1)).*sin(E(:,2) + H(2))] + [L+LL;L+LR] ;

% Create evenly spaced array
%--------------------------------------------------------------------------
Vx         = zeros(N,N);
Nh         = (N-1)/2;
[Vy,Vz]    = meshgrid(-Nh:Nh,-Nh:Nh);
V.L(:,:,1) = Vx*tan(f);
V.L(:,:,2) = Vy*tan(f);
V.L(:,:,3) = Vz*tan(f);
V.R        = V.L;

% Apply rotations and translation (according to V0)
%--------------------------------------------------------------------------
V.L = permute(V.L,[3,1,2]);
V.R = permute(V.R,[3,1,2]);

% Rotation matrices
%--------------------------------------------------------------------------
for k = 1:2
    Ry = [cos(E(k,2) + H(2)) 0 sin(E(k,2) + H(2)); 0 1 0; -sin(E(k,2) + H(2)) 0 cos(E(k,2) + H(2))]; % Around y axis
    Rz = [cos(E(k,1) + H(1)) -sin(E(k,1) + H(1)) 0; sin(E(k,1) + H(1)) cos(E(k,1) + H(1)) 0; 0 0 1]; % Around z axis
    R{k} = Rz*Ry;
end

% Apply affine transforms to retinal array
%--------------------------------------------------------------------------
for i = 1:size(V.L,2)
    for j = 1:size(V.L,3)
        V.L(:,i,j) = R{1}*(V.L(:,i,j)) + V0(1,:)';
        V.R(:,i,j) = R{2}*(V.R(:,i,j)) + V0(2,:)';
    end
end

% Return V.L and V.R
%--------------------------------------------------------------------------
V.L = permute(V.L,[2,3,1]);
V.R = permute(V.R,[2,3,1]);

end

function r = gmav_find_surface(x,n,XYZ)
% This function finds the surface first encountered by a line originating
% at the lens, passing through a virtual retinal cell (x) and continuing in
% the direction n.
%--------------------------------------------------------------------------
r = [];
for ob = 1:size(XYZ,1)                  % iterate through objects
    for s1 = 1:size(XYZ{ob,1},1)-1      % iterate through surfaces along 1st dimension
        for s2 = 1:size(XYZ{ob,1},2)-1  % iterate through surfaces along 2nd dimension
            % find vectors from retinal cell to vertices of edges
            %--------------------------------------------------------------
            v(1,:) = [XYZ{ob,1}(s1,s2),XYZ{ob,2}(s1,s2),XYZ{ob,3}(s1,s2)] - x;
            v(2,:) = [XYZ{ob,1}(s1+1,s2),XYZ{ob,2}(s1+1,s2),XYZ{ob,3}(s1+1,s2)] - x;
            v(3,:) = [XYZ{ob,1}(s1,s2+1),XYZ{ob,2}(s1,s2+1),XYZ{ob,3}(s1,s2+1)] - x;
            v(4,:) = [XYZ{ob,1}(s1+1,s2+1),XYZ{ob,2}(s1+1,s2+1),XYZ{ob,3}(s1+1,s2+1)] - x;

            % assess whether n passes through the surface
            %--------------------------------------------------------------
            M  = -[(v(1,:)' - v(2,:)'), (v(1,:)' - v(3,:)'),  -n'];
            Ma = [(v(4,:)' - v(2,:)'), (v(1,:)' - v(2,:)')];               
            Mb = [(v(4,:)' - v(3,:)'), (v(3,:)' - v(1,:)')];
            ma = [v(1,:)' - v(2,:)', v(3,:)' - v(1,:)'];
            mb = [v(4,:)' - v(1,:)', v(1,:)' - v(2,:)'];

            if rank(M) < 3
                di = [0 0 0];
            else
                SN = M\v(1,:)';
                aMax  = pinv(Ma)*(ma(:,1) + SN(2)*ma(:,2));
                bMax  = pinv(Mb)*(mb(:,1) + SN(1)*mb(:,2));
                di(1) = SN(1) >= -1/2 && SN(1) <= aMax(2);
                di(2) = SN(2) >= -1/2 && SN(2) <= bMax(2);
                di(3) = SN(3) > 0;
            end
            
            % assess whether any other more proximal surface in line of
            % sight
            %--------------------------------------------------------------
            if prod(di)
                if isempty(r)
                    r = [ob,s1,s2];
                    d = sum(v,1)/4;
                    d = d*d';
                else
                    d_ = sum(v,1)/4;
                    d_ = d_*d_';
                    if d_ < d
                        r = [ob,s1,s2];
                        d = d_;
                    end
                end
            end
        end
    end
end

end

function j = gmav_apply_lighting(r,n,XYZ,L,a,b,c)
% This function returns the amount of light reflected from a surface
% (specified in r) in a direction opposite to the vector n, in a world
% with geometries XYZ, and given a light source projecting parallel light
% along the direction L. a is ambient lighting, and b is diffuse lighting.
% c is the specular lighting component
%--------------------------------------------------------------------------

% For figure 4 of the paper, images with each lighting component were
% generated by omitting two out of three of the components:
%--------------------------------------------------------------------------
% a = 0;
% b = 0;
% c = 0;

if ~isempty(r) % There is a surface in the line of sight
    % Apply ambient lighting
    %----------------------------------------------------------------------
    j = a;
    for i = 1:size(L,1)
        LS = L(i,:);
        % Apply reflected light
        %------------------------------------------------------------------
        m = [cos(LS(1))*cos(LS(2)),sin(LS(1))*cos(LS(2)),sin(LS(2))]; % normal light vector
        
        % Check whether exposed to light
        %------------------------------------------------------------------
        x = [0,0,0];
        for k1 = 0:1
            for k2 = 0:1
                x = x + [XYZ{r(1),1}(r(2)+k1,r(3)+k2),XYZ{r(1),2}(r(2)+k1,r(3)+k2),XYZ{r(1),3}(r(2)+k1,r(3)+k2)]/4;
            end
        end
     
        s = gmav_find_surface(x,-m,XYZ);
        if isempty(s)  % nothing between surface and light source
            % find normal to surface at acute angle to n
            %--------------------------------------------------------------
            u = [XYZ{r(1),1}(r(2),r(3)),XYZ{r(1),2}(r(2),r(3)),XYZ{r(1),3}(r(2),r(3))];
            v = [XYZ{r(1),1}(r(2)+1,r(3)),XYZ{r(1),2}(r(2)+1,r(3)),XYZ{r(1),3}(r(2)+1,r(3))];
            w = [XYZ{r(1),1}(r(2),r(3)+1),XYZ{r(1),2}(r(2),r(3)+1),XYZ{r(1),3}(r(2),r(3)+1)];
            z = cross((v - u)',(w - u)')';
            z = z/sqrt(z*z');
            if z*n' > 0
                z = -z;
            end
            
            % assess whether acute angle to LS
            %--------------------------------------------------------------
            if z*m' < 0
                j = j - b*z*m';                      % add diffuse lighting
                h = (m + n)/sqrt((m + n)*(m + n)');  % find direction halfway between eye and light
                j = j + c*abs(h*z')^60;              % add specular lighting
            end
        end
    end
else
    j = 0;
    
end

end

function xyz = gmav_object(k)
% This function takes an array containing object identifier
% and returns the coordinates of the associated objects.
%--------------------------------------------------------------------------
xyz     = {};
[x,y,z] = sphere(10);

if k == 1
    Q = [0,0,0;        % Rotations
         0,0,0;
         0,0,0];
    S = [1,2,1;        % Scalings
         2,1,1;
         2,2,1];
    T = [0,0,0;        % Translations
         0,0,1;
         0,0,2];
     
elseif k == 2
    Q = [0,0,0;        % Rotations
         0,0,0;
         0,0,0];
    S = [1,2,1;        % Scalings
         1,1,1;
         1,1,1];
    T = [0,0,0;        % Translations
         0,0,1;
         1,0,0];
end

for i = 1:size(Q,1)
    [X,Y,Z] = gmav_rotate_scale_translate(Q(i,:),S(i,:),T(i,:)',x,y,z);
    xyz = [xyz;{X,Y,Z}];
end

end

function gmav_plot_eyes(q1,q2,qh,X,s)
% This function is based on that used to visualise eye-movements in Parr
% and Friston, 2018 (PMID 29407941), but is additionally equipped with head
% location and angle.
%--------------------------------------------------------------------------

r  = 3;   % Radius of head
rr = 3 + 1;
[x,y,z] = sphere;

% Plot circle representing head location, with spheres for each eye
%--------------------------------------------------------------------------
plot3(r*sin(1:360)+X(1)-rr*cos(qh),r*cos(1:360)+X(2)-rr*sin(qh),zeros(1,360)+X(3)), hold on
surf(x+r*cos(qh)-s*sin(qh)+X(1)-rr*cos(qh),y+r*sin(qh)+s*cos(qh)+X(2)-rr*sin(qh),z+X(3),'Facecolor','w','EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.7)
surf(x+r*cos(qh)+s*sin(qh)+X(1)-rr*cos(qh),y+r*sin(qh)-s*cos(qh)+X(2)-rr*sin(qh),z+X(3),'Facecolor','w','EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.7)

% Create irises
%--------------------------------------------------------------------------
a  = hgtransform;
b  = hgtransform;
xi = (x(1:5,:)); yi = y(1:5,:); zi = z(1:5,:)-exp(-4);
surf(xi,yi,zi,'Facecolor','b','EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.7,'Parent',a)
surf(xi,yi,zi,'Facecolor','b','EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.7,'Parent',b)

% ... and pupils
%--------------------------------------------------------------------------
xp = x(1:3,:); yp = y(1:3,:); zp = z(1:3,:)-2*exp(-4);
surf(xp,yp,zp,'Facecolor','k','EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.7,'Parent',a)
surf(xp,yp,zp,'Facecolor','k','EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.7,'Parent',b)

% Add lighting
%--------------------------------------------------------------------------
light

% Determine affine transformations to bring head and eyes into position
%--------------------------------------------------------------------------
Rxa = makehgtform('xrotate',q1(1));
Rxb = makehgtform('xrotate',q2(1));
Rya = makehgtform('yrotate',q1(2)-pi/2);
Ryb = makehgtform('yrotate',q2(2)-pi/2);
Rza = makehgtform('zrotate',q1(1)+qh);
Rzb = makehgtform('zrotate',q2(1)+qh);
Rta = makehgtform('translate', [s*sin(qh)+r*cos(qh)+X(1)-rr*cos(qh) -s*cos(qh)+r*sin(qh)+X(2)-rr*sin(qh) X(3)]);
Rtb = makehgtform('translate', [-s*sin(qh)+r*cos(qh)+X(1)-rr*cos(qh) +s*cos(qh)+r*sin(qh)+X(2)-rr*sin(qh) X(3)]);

% Apply affine transforms
%--------------------------------------------------------------------------
set(a,'Matrix',Rta*Rxa*Rza*Rya);
set(b,'Matrix',Rtb*Rxb*Rzb*Ryb);

end