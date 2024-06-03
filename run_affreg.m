function run_affreg(source_img, template_img, output_img_path, output_matrix_path)

    % Add SPM to the MATLAB path
    addpath(genpath('/opt/spm12'))

    % Load the images
    VF = spm_vol(source_img); % Source image volume
    VG = spm_vol(template_img); % Template image volume

    % Setup the flags for affine registration
    #flags = struct('regtype', 'affine', 'sep', 4);
    #should I smooth the images?

    % Perform affine registration
    [M, ~] = spm_affreg(VG, VF);
    #, flags)

    % Apply the transformation matrix to the source image
    VF.mat = M * VF.mat;

    % Prepare to reslice, but manage outputs manually
    reslice_flags = struct('which', 1, 'mean', false, 'interp', 1, 'prefix', '');
    spm_reslice([VG, VF], reslice_flags);

    % Load the resliced image (it should be in memory now)
    resliced_data = spm_read_vols(VF);

    % Update VF structure to save the image with the desired output filename
    VF.fname = output_img_path;

    % Write the resliced and transformed image data to disk
    spm_write_vol(VF, resliced_data);

    % Gzip the output file to save it as .nii.gz and then delete the .nii file
    gzip(output_img_path);
    delete(output_img_path);

    % Write the transformation matrix to disk
    save(output_matrix_path, 'M');

    disp(['Transformed and resliced image saved to: ', output_img_path]);
    disp(['Transformation matrix saved to: ', output_matrix_path]);
end
