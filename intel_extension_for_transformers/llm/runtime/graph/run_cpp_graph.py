import json
import os
import time
import subprocess
import pathlib
import argparse
import psutil
# args
parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument("--command",
        type=str,
        help="inference command",
        default="./build/bin/main_llama",
    )
parser.add_argument('--input_tokens', default=32, type=int)
parser.add_argument('--prompt', default=None, type=str)
parser.add_argument('--model', default="llama", type=str)
parser.add_argument('--output_tokens', default=32, type=int)
parser.add_argument('--log_file', default="log", type=str)
args = parser.parse_args()
print(args)

prompt_json = '/prompt.json'
# input prompt
current_path = os.getcwd() 
print(current_path)
working_path = os.environ.get("WORKSPACE") + "/lpot-validation/nlp-toolkit/scripts/"
print(working_path)
cores_per_instance = os.environ.get("cores_per_instance")
with open(str(working_path) + prompt_json) as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif str(args.input_tokens) in prompt_pool:
    prompt = prompt_pool[str(args.input_tokens)]
else:
    raise SystemExit('[ERROR] Plese use --prompt if want to use custom input.')


# start
total_time = 0.0
num_iter = 10
num_warmup = 4
prefix = "OMP_NUM_THREADS=%d numactl -m 0 -C 0-%d " % (int(cores_per_instance), int(cores_per_instance) - 1)
extra_cmd = " --seed 1234 --keep -1 -t 32 --repeat_penalty 1.0 --color -c %d -n %d -m %s -p \"%s\"" % (args.input_tokens + args.output_tokens + 4, args.output_tokens, args.model ,prompt)
postfix = " 2>&1 |tee %s/%s || true" % (os.environ.get("WORKSPACE"), args.log_file)
cmd =  prefix + args.command + extra_cmd + postfix
print(cmd)
for i in range(num_iter):

    memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
    print("Iteration: " + str(i), "memory used total:", memory_allocated, "GB") 
    tic = time.time()
    #os.system(cmd)
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)   # nosec
    #if process.returncode != 0:
    #    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
    toc = time.time()
    if i >= num_warmup:
        total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))
