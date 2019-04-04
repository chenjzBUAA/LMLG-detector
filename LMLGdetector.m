function [keypoints,num_keypoints] = LMLGdetectorpr(imgname)

img = imread(imgname);
data           = img;

data1          = data;
[M,N,C]        = size(data) ;
Oct            = floor(log2(min(M,N)))-4;
k              = 2^(1/3);
sigma          = 1.6*k.^(1:3);
sigma          = sigma';

%------ step 1: calculate the response in scales

for iio = 1:Oct    
    for scale_index = 1:3
        scale = sigma(scale_index);
        [Res{iio}(:,:,scale_index)]  = LMLGFilterRes(data,scale);
    end
    data = uint8(downsampling(data));
end
%------ step 2 detect the local maximum in both the scale and the space
data   = data1;
oframe = [];
omin   =  0;
for iio = 1:Oct
    frame_scale = [];
    for jj = 1:3       
        s_L          = sigma(jj);    
        w            = ceil(4*s_L);
        data_smooth  = [];
        data_smooth  = imfilter(double(data),fspecial('gaussian',2*w+1,s_L),'replicate');
        % 2.1 choose the threshold for each scale by the following equation
       
        sigma1 = 2^(iio-1+omin) * sigma(jj) ;  
        
        rank   = 0.93;
        thres  = abs(Res{iio}(:,:,jj));
        thres  = sort( thres(:) , 'ascend' );
        thres  = thres( ceil( length( thres ) *rank ) );
        range               = 2*round(s_L)+1;     
        maskss              = sum(sum(ones(range,range)));
        Reptemp             = Res{iio}(:,:,jj);
        X11                 = ordfilt2(Reptemp,maskss,ones(range,range));
        X22                 = ordfilt2(Reptemp,1,ones(range,range));  

        Max_p               = Reptemp>=X11&Reptemp>thres;
        Min_p               = Reptemp<=X22&-Reptemp>thres; 

        M_p=Max_p;
        [row col]=find(M_p==1);
        pointsp=[row col];
        M_p=Min_p;
        [row col]=find(M_p==1);
        pointsn=[row col];

        points = [pointsp;pointsn];
        %------- step 3 refine the keypoints---MINIMUM
        frames              = refine_keypoints(double(data_smooth),points);
        %step 4. calculate the affine parameter for each keypoint.
        frames_affine       = affine_parameter_keypoints(double(data_smooth),frames,sigma1);
              
        frame_scale         = [frame_scale;frames_affine];
    end
    if length(frame_scale)~=0
        frame_scale(:,1:2)=frame_scale(:,1:2)*2^(iio-1+omin);
        oframe  = [oframe; frame_scale ] ;
    end
    data = imresize(data,0.5);
end

%output keypoints
keypoints = oframe;
num_keypoints = size(oframe,1);

% calculate the response of M in scale S.
function [Result] = LMLGFilterRes(data,scale)
data=double(data);

weight_mask = -fspecial('log',2*floor(2.6*scale)+1,scale);
[winm winn] = size(weight_mask);
Result1=conv2(data,weight_mask,'valid');
Result1=padarray(Result1,[floor(winm/2) floor(winm/2)]);

h = floor(scale*1.5);
smooth  = imfilter(data,fspecial('gaussian',2*h+1,scale/2),'replicate');

win=2*floor(2.6*scale)+1;
val=ceil(win*win/2);
Result2=(smooth-ordfilt2(smooth,val,ones(win,win),'symmetric'));

Result=((Result1>10^-5).*Result1).*((Result2>10^-5).*Result2)+(-(Result1<-10^-5).*Result1).*((Result2<-10^-5).*Result2);


function[sls]=downsampling(vect)
%image_downsampling
vect=double(vect);
sls=...
   0.25*(...
         vect(2:2:end-1,2:2:end-1)+...
         0.5*(vect(1:2:end-2,2:2:end-1)+vect(3:2:end,2:2:end-1)+vect(2:2:end-1,1:2:end-2)+vect(2:2:end-1,3:2:end))+...
         0.25*(vect(1:2:end-2,1:2:end-2)+vect(3:2:end,1:2:end-2)+vect(3:2:end,3:2:end)+vect(1:2:end-2,3:2:end))...
         );



function frames = refine_keypoints(data_smooth,spoints)  
points_num = size(spoints,1);
frames         = [];
r              = 10;
[mm nn]        = size(data_smooth);

for kk = 1:points_num
    % calculate the curvature for each local keypoints  
    if (round(spoints(kk,1))>1 && round(spoints(kk,1))<mm && round(spoints(kk,2))>1 && round(spoints(kk,2))<nn)
        patch_D        = data_smooth(round(spoints(kk,1))+(-1:1),round(spoints(kk,2))+(-1:1));
        Dyy            = (patch_D(3,2) + patch_D(1,2)- 2.0 * patch_D(2,2)) ;
        Dxx            = (patch_D(2,3) + patch_D(2,1)- 2.0 *  patch_D(2,2)) ;       
        Dxy            = 0.25 * ( patch_D(3,3) + patch_D(1,1) - patch_D(1,3) - patch_D(3,1) ) ; 
        detM           = (Dxx*Dyy - Dxy*Dxy);
        if abs(detM)>0.001
            score = (Dxx+Dyy)*(Dxx+Dyy) / detM;
            if abs(score)<(r+1)*(r+1)/r
                a = 1;
                b = 1;
                c = 1;
                frames = [frames;spoints(kk,:)];
            end
        end
    end
end

function [frames] = affine_parameter_keypoints(data_smooth,spoints,sigma_ori) 
points_num = size(spoints,1);
frames         = [];
for kk = 1:points_num
    a          = 1 / (2*sigma_ori)^2;
    b          = 0;
    c          = 1 / (2*sigma_ori)^2;
    frames     = [frames;spoints(kk,2) spoints(kk,1) a b c sigma_ori];
end


