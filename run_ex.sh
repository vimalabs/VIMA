ckpt_path="/home/rlcorrea/CSE574_project_vima/model_weights/2M.ckpt"
device="cuda:0"
eval_level="placement_generalization"
task="rotate"
python3 ./scripts/example_mp4.py --ckpt=${ckpt_path} --device=${device} --partition=${eval_level} --task=${task}
