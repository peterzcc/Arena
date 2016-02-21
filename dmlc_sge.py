#!/usr/bin/env python
"""
DMLC submission script, local machine version
"""

import argparse
import sys
import os
import subprocess
import tracker
import logging
import signal
keepalive = """
nrep=0
rc=254
while [ $rc -eq 254 ];
do
    export DMLC_NUM_ATTEMPT=$nrep
    %s
    rc=$?;
    nrep=$((nrep+1));
done
"""

class SgeLauncher(object):
    def __init__(self, args, unknown):
        self.args = args
        self.cmd = ' '.join(args.command) + ' ' + ' '.join(unknown)
        self.unknown = unknown
    def submit(self):
        def sge_submit(nworker, nserver, pass_envs):
            """
              customized submit script, that submit nslave jobs, each must contain args as parameter
              note this can be a lambda function containing additional parameters in input
              Parameters
                 nworker number of slave process to start up
                 nserver number of server nodes to start up
                 pass_envs enviroment variables to be added to the starting programs
            """
            pass_envs['DMLC_ROLE'] = 'worker'
            env_arg = ','.join(['%s=\"%s\"' % (k, str(v)) for k, v in pass_envs.items()])
            cmd = 'qsub -cwd -t 1-%d -S /bin/bash' % (nworker)
            if self.args.queue != 'default':
                cmd += ' -q %s' % self.args.queue
            cmd += ' -N %s ' % self.args.jobname
             
            #cmd += ' -e %s -o %s' % (self.args.logdir, self.args.logdir)
            #cmd += ' -pe orte %d' % (self.args.vcores)
            cmd += '-o otest.txt -j y'
            cmd += ' -v %s,PATH=${PATH}:.' % env_arg
            cmd += ' %s %s' % (self.args.runscript,' '.join(self.args.command) + ' ' + ' '.join(self.unknown))

            pass_envs['DMLC_ROLE'] = 'server'
            env_arg = ','.join(['%s=\"%s\"' % (k, str(v)) for k, v in pass_envs.items()])
            cmd += '&& qsub -cwd -t 1-%d -S /bin/bash' % (nserver)
            cmd += ' -q %s' % self.args.queue
            cmd += ' -N %s ' % self.args.jobname
            cmd += ' -e %s -o %s' % (self.args.logdir, self.args.logdir)
            cmd += ' -v %s,PATH=${PATH}:.' % env_arg
            cmd += ' %s %s' % (self.args.runscript,' '.join(self.args.command) + ' ' + ' '.join(self.unknown))
            
            logging.info(cmd)
            subprocess.check_call(cmd, shell = True)
            logging.info('Waiting for the jobs to get up...')
        return sge_submit

    def run(self):
        tracker.config_logger(self.args)
        tracker.submit(self.args.num_workers,
                       self.args.num_servers,
                       fun_submit = self.submit(),
                       pscmd = self.cmd)

def main():
    parser = argparse.ArgumentParser(description='DMLC script to submit dmlc job using Sun Grid Engine')
    parser.add_argument('-n', '--num_workers', required=True, type=int,
                        help = 'number of worker proccess to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help = 'number of server nodes to be launched')
    parser.add_argument('-q', '--queue', default='*.q@client108,*.q@client109,*.q@client110', type=str,
                        help = 'the queue we want to submit the job to')
    parser.add_argument('--log-level', default='INFO', type=str,
                        choices=['INFO', 'DEBUG'],
                        help = 'logging level')
    parser.add_argument('--log-file', type=str,
                        help = 'output log to the specific log file')
    parser.add_argument('--logdir', default='auto', help = 'customize the directory to place the SGE job logs')
    parser.add_argument('-hip', '--host_ip', default='auto', type=str,
                        help = 'host IP address if cannot be automatically guessed, specify the IP of submission machine')
    parser.add_argument('--vcores', default = 1, type=int,
                        help = 'number of vcpores to request in each mapper, set it if each dmlc job is multi-threaded')
    parser.add_argument('--jobname', default='auto', help = 'customize jobname in tracker')
    parser.add_argument('command', nargs='+',
                        help = 'command for dmlc program')
    args, unknown = parser.parse_known_args()

    if args.jobname == 'auto':
        args.jobname = ('dmlc%d.' % args.num_workers) + args.command[0].split('/')[-1];
    if args.logdir == 'auto':
        args.logdir = args.jobname + '.log'

    if os.path.exists(args.logdir):
        if not os.path.isdir(args.logdir):
            raise RuntimeError('specified logdir %s is a file instead of directory' % args.logdir)
    else:
        os.mkdir(args.logdir)
    if args.num_servers is None:
        args.num_servers = args.num_workers

    runscript = '%s/rundmlc.sh' % args.logdir
    fo = open(runscript, 'w')
    fo.write('#!/bin/bash\n#$ -S /bin/bash\n. /project/dygroup2/czeng/venv/bin/activate\n')
    fo.write('export DMLC_TASK_ID=${SGE_TASK_ID}\n')
    #fo.write(' '.join(args.command) + ' ' + ' '.join(unknown)+"\n")
    fo.write('\"$@\"\n')
    fo.close()
    args.runscript = runscript
    launcher = SgeLauncher(args, unknown)
    launcher.run()
def signal_handler(signal, frame):
    logging.info('Stop luancher')
    sys.exit(0)
if __name__ == '__main__':
    #signal.signal(signal.SIGINT, signal_handler)
    main()
