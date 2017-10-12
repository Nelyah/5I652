baseDir = 'pictures/Scene/'
baseDirDes = 'descriptors_files/'
nomDico = 'visual_dictionary.txt'

clusters = load(nomDico);

%[ patchmin ] = visuDico( nomDico ,  baseDir , baseDirDes);
%save('visu_dico.mat', 'patchmin');
struct = load('visu_dico.mat');
patchmin = struct.patchmin

[ I , nomim , sifts] = randomImageDes( baseDir , baseDirDes);
sifts = double(sifts);
[bow, nc] = computeBow(sifts', clusters);
visuBoW(I,patchmin,bow,nc,nomim)

%[ patchmin ] = visuDico( nomDico ,  baseDir , baseDirDes)