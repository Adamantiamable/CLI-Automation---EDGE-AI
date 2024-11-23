from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

command_definitions = """Utilisation : cp [OPTION]... [-T] SOURCE DEST
         ou : cp [OPTION]... SOURCE... RÉPERTOIRE
         ou : cp [OPTION]... -t RÉPERTOIRE SOURCE...
Copier la SOURCE vers DEST ou plusieurs SOURCEs vers RÉPERTOIRE.

Les arguments obligatoires pour les options longues le sont aussi pour les
options courtes.
  -a, --archive                identique à -dR --preserve=all
      --attributes-only        ne pas copier les données du fichier, seulement
                                 les attributs
      --backup[=CONTRÔLE]      archiver chaque fichier de destination
  -b                           comme --backup mais n'accepte pas d'argument
      --copy-contents          copier le contenu des fichiers spéciaux en mode
                                 récursif
  -d                           identique à --no-dereference --preserve=links
      --debug                  expliquer comme le fichier est copié. Implique -v
  -f, --force                  si un fichier de destination existe et ne peut
                                 être ouvert, alors le supprimer et réessayer
                                 (cette option est ignorée si l'option -n est
                                 aussi utilisée)
  -i, --interactive            demander confirmation avant d'écraser (annule
                                 l'option -n précédente)\n
Utilisation : ls [OPTION]... [FICHIER]...

Les arguments obligatoires pour les options longues le sont aussi pour les
options courtes.
  -a, --all                  ne pas ignorer les entrées débutant par .
  -A, --almost-all           ne pas inclure . ou .. dans la liste
      --author               avec -l, afficher l'auteur de chaque fichier
  -b, --escape               afficher les caractères non graphiques avec des
                               protections selon le style C
      --block-size=TAILLE    avec -l, dimensionner les tailles selon TAILLE avant
                             de les afficher. Par exemple, « --block-size=M ».
                             Consultez le format de TAILLE ci-dessous

  -B, --ignore-backups       ne pas inclure les entrées terminées par ~ dans la liste
  -c                         avec -lt : afficher et trier selon ctime (date de
                             dernier changement des informations d'état du fichier) ;
                             avec -l : afficher ctime et trier selon le nom ;
                             autrement : trier selon ctime, le plus récent en
                             premier\n\n
sudo – exécute une commande en tant qu'un autre utilisateur

usage: sudo -h | -K | -k | -V
usage: sudo -v [-ABkNnS] [-g group] [-h host] [-p prompt] [-u user]
usage: sudo -l [-ABkNnS] [-g group] [-h host] [-p prompt] [-U user]
            [-u user] [command [arg ...]]
usage: sudo [-ABbEHkNnPS] [-r role] [-t type] [-C num] [-D directory]
            [-g group] [-h host] [-p prompt] [-R directory] [-T timeout]
            [-u user] [VAR=value] [-i | -s] [command [arg ...]]
usage: sudo -e [-ABkNnS] [-r role] [-t type] [-C num] [-D directory]
            [-g group] [-h host] [-p prompt] [-R directory] [-T timeout]
            [-u user] file ...

Options:
  -A, --askpass                 utiliser un programme adjoint pour demander le
                                mot de passe
  -b, --background              exécuter la commande en arrière-plan
  -B, --bell                    ring bell when prompting
  -C, --close-from=num          fermer tous les descripteurs de fichiers >= n°
  -D, --chdir=directory         change the working directory before running
                                command
  -E, --preserve-env            préserver l'environnement de l'utilisateur en
                                exécutant la commande
      --preserve-env=list       preserve specific environment variables
  -e, --edit                    éditer les fichiers au lieu d'exécuter une
                                commande
"""

system_prompt = """You are an expert in using the Linux command line. You are given a question and a set of possible commands. 
Based on the question, you will need to suggest one or more commands to achieve the purpose. Those commands can be used sequentially. 
Only use the information provided in the commands. 
If none of the commands can be used, point it out. If the given question lacks the parameters required by the function, 
fill them with generic variables. Only output the commands in their squential order.
Make sure the subsequent use of the commands make sense and can be executed by the user. 
ONLY WRITE A SINGLE SENTENCE. DO NOT EXPLAIN YOUR ANSWER. Your life depends on this. 
Here is a list of commands with their documentation.\n\n{commands}\n""".format(commands=command_definitions)

user_prompt = "I want to move many files from one directory to another. How can I do that?"

prompt = (
    system_prompt + '\n' + user_prompt
)

tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

print('Generating...')
output_tokens = model.generate(**tokens, max_new_tokens=50, temperature=0.7, repetition_penalty=1.1)

prompt_length = tokens['input_ids'].shape[1]

print('Decoding...')
output_text = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)

print(output_text)