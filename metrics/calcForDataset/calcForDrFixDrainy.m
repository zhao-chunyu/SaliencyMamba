function calcForDrFixDrainy(root_path, root_path2, root_path3)
    warning('off');

    % test data_name
    load(root_path3);

    % save metrics
    auc_b_scores = [];
    auc_j_scores = [];
    NSS_scores = [];

    CC_scores = [];
    KLD_scores = [];
    Sim_scores = [];


    %% start computing
    tic;
    for i=1:size(test,1)
    %% Compute ---> CC, SIM, Kld
    % p--->predict map   f--->fixation map

        pre_img_path = [[root_path, 'pre/'],test(i, 1:9), '.mat'];
        p = getfield(load(pre_img_path), 'pre');
        p = double(p);
        p = (p-min(p(:)))./(max(p(:))-min(p(:)));
        saliency_map = p;

        tar_img_path = [[root_path, 'tar/'],test(i, 1:9), '.mat'];
        f = getfield(load(tar_img_path), 'tar');
        f = double(f);
        f = (f-min(f(:)))./(max(f(:))-min(f(:)));
        fix_saliency_map = f;

        
    %% Compute ---> AUC_B, AUC_J, NSS
        video_index = str2num(test(i,1:2));
        frame_index = str2num(test(i,4:9));
        fix_path = [root_path2, num2str(video_index),'.mat'];
        load(fix_path);
        frame_fixdata=fixdata{frame_index,1};
        fix_map=zeros(720,1280);
        x=frame_fixdata(:,4);
        y=frame_fixdata(:,3);
        xy=[x y];

        for j=1:length(x)
            temp_x = x(j);
            temp_y = y(j);
            if temp_x<721 && temp_x>0 && temp_y<1281 &&temp_y>0
                fix_map(temp_x,temp_y)=1;
            end
        end
        

    %% metrics
        [auc_b_score,tp_b,fp_b] = AUC_Borji(saliency_map, fix_map);
        auc_b_scores(i) = mean(auc_b_score);
    
        [auc_j_score,tp_j,fp_j] = AUC_Judd(saliency_map, fix_map);
        auc_j_scores(i) = mean(auc_j_score);
    
        NSS_score = NSS(saliency_map, fix_map);
        NSS_scores(i) = mean(NSS_score);
    
        CC_score = CC(saliency_map, fix_saliency_map);
        CC_scores(i) = mean(CC_score);
    
        KLD_score = KLdiv(saliency_map, fix_saliency_map);
        KLD_scores(i) = mean(KLD_score);
        
        Sim_score = similarity(saliency_map, fix_saliency_map);
        Sim_scores(i) = mean(Sim_score);

            
        if ( mod(i, 50) == 0 )
            a = zeros(6,1);
            a(1) = mean(auc_b_scores(:));
            a(2) = mean(auc_j_scores(:));
            a(3) = mean(NSS_scores(:));
            a(4) = mean(CC_scores(:));
            a(5) = mean(Sim_scores(:));
            a(6) = mean(KLD_scores(:));
            disp(sprintf('Iter %5d --> AUC_borji %.4f | AUC_judd %.4f | NSS %.4f | CC %.4f | SIM %.4f | KLD %.4f | #time %.4fs',i,a(1),a(2),a(3),a(4),a(5),a(6),toc));
            tic;
        end
    end

    a = zeros(6,1);
    a(1) = mean(auc_b_scores(:));
    a(2) = mean(auc_j_scores(:));
    a(3) = mean(NSS_scores(:));
    a(4) = mean(CC_scores(:));
    a(5) = mean(Sim_scores(:));
    a(6) = mean(KLD_scores(:));
    disp(sprintf('Ending Computing ---> AUC_borji %.4f | AUC_judd %.4f | NSS %.4f | CC %.4f | SIM %.4f | KLD %.4f',a(1),a(2),a(3),a(4),a(5),a(6)));
    save([root_path, 'metrics.mat'], 'auc_b_scores', 'auc_j_scores', 'NSS_scores', 'CC_scores', 'Sim_scores', 'KLD_scores');