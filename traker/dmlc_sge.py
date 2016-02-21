#!/usr/bin/env python
"""
DMLC submission script, HKUST cluster2.cse.ust.hk version
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


def gen_run_script(args, unknown):
    runscript = '%s/rundmlc.sh' % args.logdir
    fo = open(runscript, 'w')
    fo.write('#!/bin/bash\n')
    fo.write('#$ -S /bin/bash\n')
    if args.activate_cmd is not None:
        fo.write("%s\n" % args.activate_cmd)
    fo.write("#$ -wd %s\n" % args.working_dir)
    fo.write('source ~/.bashrc\n')
    fo.write('export DMLC_TASK_ID=${SGE_TASK_ID}\n')
    fo.write(' '.join(args.command) + ' ' + ' '.join(unknown) + "\n")
    fo.close()
    args.runscript = runscript
    return runscript

def submit_worker(num, node, pass_envs, args):
    pass_envs['DMLC_ROLE'] = 'worker'
    env_arg = ','.join(['%s=\"%s\"' % (k, str(v)) for k, v in pass_envs.items()])
    cmd = 'qsub -cwd -S /bin/bash'
    cmd += ' -q %s' % node
    cmd += ' -N %s-worker-%d' % (args.jobname, num)
    cmd += ' -o %s -j y' % args.log_file
    cmd += ' -v %s,PATH=${PATH}:.' % env_arg
    cmd += ' %s' % (args.runscript)
    logging.info(cmd)
    subprocess.check_call(cmd, shell=True)


def submit_server(num, node, pass_envs, args):
    pass_envs['DMLC_ROLE'] = 'server'
    env_arg = ','.join(['%s=\"%s\"' % (k, str(v)) for k, v in pass_envs.items()])
    cmd = 'qsub -cwd -S /bin/bash'
    cmd += ' -q %s' % node
    cmd += ' -N %s-server-%d ' % (args.jobname, num)
    cmd += ' -e %s -o %s' % (args.logdir, args.logdir)
    cmd += ' -v %s,PATH=${PATH}:.' % env_arg
    cmd += ' %s' % (args.runscript)
    logging.info(cmd)
    subprocess.check_call(cmd, shell=True)


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
            serverq_l = self.args.server_queue.split(',')
            workerq_l = self.args.worker_queue.split(',')
            print serverq_l, self.args.server_queue
            for i in range(nworker):
                submit_worker(num=i, node=workerq_l[i % len(workerq_l)], pass_envs=pass_envs,
                              args=self.args)
            for i in range(nserver):
                submit_server(num=i, node=serverq_l[i % len(serverq_l)], pass_envs=pass_envs,
                              args=self.args)
            logging.info('Waiting for the jobs to get up...')

        return sge_submit

    def run(self):
        tracker.config_logger(self.args)
        tracker.submit(self.args.num_workers,
                       self.args.num_servers,
                       fun_submit=self.submit(),
                       pscmd=self.cmd)


def main():
    parser = argparse.ArgumentParser(
        description='DMLC script to submit dmlc job using Sun Grid Engine')
    parser.add_argument('-n', '--num_workers', required=True, type=int,
                        help='number of worker proccess to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help='number of server nodes to be launched')
    parser.add_argument('-wd', '--working-dir', type=str,
                        help='working directory')
    parser.add_argument('-serverq', '--server-queue', default='all.q@client108,all.q@client109',
                        type=str,
                        help='the queue we want to submit our server jobs to')
    parser.add_argument('-workerq', '--worker-queue', default='all.q@client110,all.q@client111',
                        type=str,
                        help='the queue we want to submit our worker jobs to')
    parser.add_argument('--log-level', default='INFO', type=str,
                        choices=['INFO', 'DEBUG'],
                        help='logging level')
    parser.add_argument('--log-file', default='otest.out', type=str,
                        help='output log of workers to the specific log file')
    parser.add_argument('--logdir', default='auto',
                        help='customize the directory to place the SGE job logs')
    parser.add_argument('-hip', '--host_ip', default='auto', type=str,
                        help='host IP address if cannot be automatically guessed, specify the IP of submission machine')
    parser.add_argument('--activate-cmd', default=None, type=str,
                        help='activation script of the running environment')
    parser.add_argument('--jobname', default='auto', help='customize jobname in tracker')
    parser.add_argument('command', nargs='+',
                        help='command for dmlc program')
    args, unknown = parser.parse_known_args()

    if args.jobname == 'auto':
        args.jobname = ('dmlc%d.%d' % (args.num_workers, args.num_servers)) + \
                       args.command[0].split('/')[-1]
    if args.logdir == 'auto':
        args.logdir = args.jobname + '.log'

    if os.path.exists(args.logdir):
        if not os.path.isdir(args.logdir):
            raise RuntimeError('specified logdir %s is a file instead of directory' % args.logdir)
    else:
        os.mkdir(args.logdir)
    if args.num_servers is None:
        args.num_servers = args.num_workers

    gen_run_script(args, unknown)
    launcher = SgeLauncher(args, unknown)
    launcher.run()


if __name__ == '__main__':
    main()
