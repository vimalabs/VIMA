ckpt_path="/home/rlcorrea/CSE574_project_vima/model_weights/2M.ckpt"
device="cuda:0" 
#task4 will be 
eval_level="placement_generalization"
task="rearrange"
#task5 will be 
#eval_level="combinatorial_generalization"
taks="sweep_without_exceeding"
#python3 ./scripts/example_mp4.py --ckpt=${ckpt_path} --device=${device} --partition=${eval_level} --task=${task}
python3 ./scripts/example_headless.py --ckpt=${ckpt_path} --device=${device} --partition=${eval_level} --task=${task}
