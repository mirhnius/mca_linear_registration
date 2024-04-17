function run_affreg(source_img, template_img, output_path)
    % Add SPM to the MATLAB path
    addpath('/opt/spm12');

    % Load the images
    VF = spm_vol(source_img); % Source image volume
    VG = spm_vol(template_img); % Template image volume

    % Setup the flags for affine registration
    #flags = struct('regtype', 'affine', 'sep', 4);

    % Perform affine registration
    [M, ~] = spm_affreg(VG, VF);

    % Apply the transformation matrix to the source image
    VF.mat = M*VF.mat;
    spm_reslice([VG,VF],struct('which',1));
    transformed_VF = spm_create_vol(VF);
    transformed_VF.fname = output_path;

    % Write the transformed image to disk
    spm_write_vol(transformed_VF, spm_read_vols(VF));

    disp(['Transformed image saved to: ', output_path]);
end
