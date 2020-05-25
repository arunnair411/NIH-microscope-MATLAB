%% Function to call save from within a parfor loop
function parsave(file_name, file_name_idx, input_data, target_output)
    save(file_name, 'file_name_idx', 'input_data', 'target_output');
end