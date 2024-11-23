import subprocess

examples = [
    ["ls", []],
    ["mv", ["truc.txt", "test"]],
    ["code", []],
    ["cp", ["README.md", "test"]]
]

def format_example(example):
    return_list = [example[0]]
    for argument in example[1]:
        return_list.append(argument)
    return return_list

nb_example_to_try = 1
subprocess_output = subprocess.run(
    format_example(examples[nb_example_to_try]),
    capture_output=True
)

if subprocess_output.returncode == 0:
    print(subprocess_output.stdout)
    print("Command executed successfuly")
else:
    print('stdout:')
    print(subprocess_output.stdout)
    print('stderr:')
    print(subprocess_output.stderr)