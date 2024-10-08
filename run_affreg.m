function run_affreg(source_img, template_img, output_img_path, output_matrix_path)
    % Add SPM to the MATLAB path
    addpath(genpath('/opt/spm12'))

    % Load the images
    VF = spm_vol(source_img); % Source image volume
    VG = spm_vol(template_img); % Template image volume

    % Perform affine registration
    [M, ~] = spm_affreg(VG, VF);

    % Apply the transformation matrix to a copy of the source images transformation matrix
    VF.mat = M * VF.mat;

    % Prepare flags for reslicing
    reslice_flags = struct('which', 0, 'mean', false, 'interp', 1, 'prefix', '');

    % Call spm_reslice with which=0 to perform reslicing in memory
    spm_reslice([VG, VF], reslice_flags);

    % Read the resliced data from the transformed image
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
