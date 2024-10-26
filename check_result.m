
patch_num=2; % ranged from 0 to 9
h5_in='data/hologic_sim_val.h5';
uncorrected=h5read(h5_in, ['/image_', num2str(patch_num)]);
label=h5read(h5_in, ['/label_', num2str(patch_num)]);
h5_out='results/hologic_sim_model_our_3/test.h5';
result=h5read(h5_out, ['/output_', num2str(patch_num)]);  % 2,6,7,8
slice_num=29;
imshow([uncorrected(:,:,slice_num),result(:,:,slice_num),label(:,:,slice_num)], [100,300])
