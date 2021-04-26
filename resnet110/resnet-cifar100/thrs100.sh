#!/bin/bash
#SBATCH -n 10
#SBATCH -A research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=3000

#SBATCH --time=4-00:00:00

#SBATCH --mail-type=END



module add cuda/10.0

module add cudnn/7.6-cuda-10.0 
##python train.py 0 model  model log.out 0
echo "Train completed"

echo "Test noise series are impulse_noise.npy, speckle_noise.npy, gaussian_noise.npy, shot_noise.npy"

 
python test_noise.py model.mdlpkl  0  log.out
echo "Test noise completed"

 
echo "Test_fgsm r=1"
python test_fgsm.py model.mdlpkl  0  log.out 1
 
echo "Test_fgsm r=2"
python test_fgsm.py model.mdlpkl  0  log.out 2

 
echo "Test_fgsm r=4"
python test_fgsm.py model.mdlpkl  0  log.out 4


echo "Test_ifgsm r=1"
python test_ifgsm.py model.mdlpkl  0  log.out 1

echo "Test_ifgsm r=2"
python test_ifgsm.py model.mdlpkl  0  log.out 2


echo "Test_ifgsm r=4"
python test_ifgsm.py model.mdlpkl  0  log.out 4


echo "Test_pgd r=1"
python test_pgd.py model.mdlpkl  0  log.out 1
 
echo "Test_pgd r=2"
python test_pgd.py model.mdlpkl  0  log.out 2
 
echo "Test_pgd r=4"
python test_pgd.py model.mdlpkl  0  log.out 4
 
