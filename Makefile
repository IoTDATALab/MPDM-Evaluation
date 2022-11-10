# 同步代码要使用WSL的rsync!

MINICONDA_PATH := "~/miniconda3"
PYTHON_PATH := "~/miniconda3/bin/python"


NODE := node-5
TEST := "tests/tests.yaml"


#Linux stuff
PYTHON = python3
SYNC_CMD = rsync -avp -r -P -e 'ssh -F ./ssh/ssh_config -i ./ssh/exps_id_rsa '

SSH_CMD = ssh -F ./ssh/ssh_config -i ./ssh/exps_id_rsa 

ssh-public-key: ./ssh/exps_id_rsa.pub

./ssh/exps_id_rsa.pub: 
	@ssh-keygen -t rsa -P '' -f ./ssh/exps_id_rsa;

net_config:	ssh-public-key
	@echo "Make password free configuration in ${NODE}";
	@cat ./ssh/exps_id_rsa.pub | ${SSH_CMD} ${NODE} "umask 0600; mkdir -p .ssh ; cat >> .ssh/authorized_keys" ;
	
install_python: 
	@echo "Install Miniconda3 on ${NODE}";
	@${SSH_CMD} ${NODE} bash -c "'wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda3.sh;bash ~/miniconda3.sh -b -u -p ${MINICONDA_PATH};rm ~/miniconda3.sh'";

clean_python:
	@echo "Clean Miniconda3 on ${NODE}";
	@{SSH_CMD} ${NODE} bash -c "'rm -rf ${MINICONDA_PATH}'";

install_requirements: send_code
	@echo "Install python modules on ${NODE}";
	@${SSH_CMD} ${NODE} bash -c  "'source ${MINICONDA_PATH}/bin/activate;pushd ~/exps;bash script/env.sh;popd'";

send_code:
	@echo "Transfer source files to ${NODE}";
	@${SYNC_CMD} --exclude 'plot/' --exclude '.git/'  . ${NODE}:~/exps;

run_experiments: send_code
	@echo "Run experiments on ${NODE}"
	@${SSH_CMD} ${NODE} bash -c  "'source ${MINICONDA_PATH}/bin/activate;pushd ~/exps; ${PYTHON_PATH} -W ignore main.py --path=${TEST};popd'"

stop_experiments: send_code
	@echo "Stop experiments on ${NODE}"
	@${SSH_CMD} ${NODE} bash -c  "'pushd ~/exps; bash script/kill.sh;popd'"

collect_results:
	@echo "Collect results on ${NODE}"
	@${SYNC_CMD} ${NODE}:~/exps/plot/results/ ./plot/results;
	
clean_code:
	@echo "Clean experiments on ${NODE}";
	@${SSH_CMD} ${NODE} bash -c "'rm -rf ~/exps'";

clean: clean_code clean_python