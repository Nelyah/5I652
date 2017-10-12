baseDir = 'pictures/Scene/'
baseDirDes = 'descriptors_files/'
nomDico = 'visual_dictionary.txt'
baseDirBoW = 'bow_files/'


clusters = load(nomDico);

struct = load('visu_dico.mat');
patchmin = struct.patchmin;

clusters = load(nomDico);
cate = categories();
pas = 1;

% For each category
for index = 1:15
    cat = cate{index};

    % CREATION DU DOSSIER DESCRIPTEURS POUR LA CLASSE SI INEXISTANT
    % Save the BoW in path category
    pathcat = strcat(baseDirBoW,cat,'/');
    if(exist(pathcat)==0)
        mkdir(pathcat);
    end
    % BoW for the whole category
    %bow_cate = [];

    % Path image
    path = strcat(baseDir,cat,'/');
    listima=dir([path '*.jpg'] );
    n=length(listima);
    
    % For each image in directory
    for num = 1:n
        if(num<10)
            nom = strcat('/image_000',num2str(num));
        elseif(num<100)
            nom = strcat('/image_00',num2str(num));
        else
            nom = strcat('/image_0',num2str(num));
        end

        nomim = strcat(baseDir,cat,nom,'.jpg');
        I = imread(nomim);

        nomim = strcat(cat,'-',num2str(num));

        nomdes = strcat(baseDirDes,cat,nom,'.mat');
        load(nomdes);
        
        sifts = double(sifts);
        [bow, nc] = computeBow(sifts', clusters);
        %bow_cate = [bow_cate bow];
        
        % Save the bow_cate
        filename_bow = '';
        filename_bow = [pathcat, nom, '.mat'];
        save(filename_bow, 'bow');
    end
    
    
end

































%[ I , nomim , sifts] = randomImageDes( baseDir , baseDirDes);
%sifts = double(sifts);
%[bow, nc] = computeBow(sifts', clusters);
%visuBoW(I,patchmin,bow,nc,nomim)

%[ patchmin ] = visuDico( nomDico ,  baseDir , baseDirDes)